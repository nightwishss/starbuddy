#!/usr/bin/env python3
"""Compare raw CPU and QNN outputs for identical MediaPipe TFLite inputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2


def make_interpreter(interpreter_class, model: Path, delegates=None):
    instance = interpreter_class(str(model), experimental_delegates=delegates)
    instance.allocate_tensors()
    return instance


def deterministic_input(detail: dict, image_path: Path | None, model_name: str) -> np.ndarray:
    shape = tuple(int(value) for value in detail["shape"])
    dtype = detail["dtype"]
    count = int(np.prod(shape))
    if image_path is not None:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"cannot read {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = shape[1:3]
        if model_name == "pose_detector.tflite":
            scale = min(width / image.shape[1], height / image.shape[0])
            resized = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)))
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            top = (height - resized.shape[0]) // 2
            left = (width - resized.shape[1]) // 2
            canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
            value = (canvas.astype(np.float32) - 127.5) / 127.5
        else:
            resized = cv2.resize(image, (width, height))
            value = resized.astype(np.float32) / 255.0
        return value.reshape(shape).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        return np.linspace(-1.0, 1.0, count, dtype=np.float32).reshape(shape).astype(dtype)
    info = np.iinfo(dtype)
    return np.linspace(info.min, info.max, count).reshape(shape).astype(dtype)


def invoke(interpreter, tensor: np.ndarray) -> tuple[float, list[np.ndarray]]:
    input_detail = interpreter.get_input_details()[0]
    interpreter.set_tensor(input_detail["index"], tensor)
    before = time.perf_counter()
    interpreter.invoke()
    elapsed = (time.perf_counter() - before) * 1000
    return elapsed, [interpreter.get_tensor(item["index"]).astype(np.float64)
                     for item in interpreter.get_output_details()]


def compare(cpu: np.ndarray, qnn: np.ndarray) -> dict:
    delta = np.abs(cpu - qnn).reshape(-1)
    scale = np.maximum(np.abs(cpu).reshape(-1), 1e-6)
    return {
        "shape": list(cpu.shape),
        "mean_absolute_error": round(float(delta.mean()), 8),
        "p95_absolute_error": round(float(np.percentile(delta, 95)), 8),
        "max_absolute_error": round(float(delta.max()), 8),
        "mean_relative_error": round(float((delta / scale).mean()), 8),
        "cpu_range": [round(float(cpu.min()), 6), round(float(cpu.max()), 6)],
        "qnn_range": [round(float(qnn.min()), 6), round(float(qnn.max()), 6)],
    }


def run(interpreter_class, delegate_factory, model: Path, image_path: Path | None) -> dict:
    cpu = make_interpreter(interpreter_class, model)
    qnn = make_interpreter(interpreter_class, model, delegate_factory())
    tensor = deterministic_input(cpu.get_input_details()[0], image_path, model.name)
    cpu_ms, cpu_outputs = invoke(cpu, tensor)
    qnn_ms, qnn_outputs = invoke(qnn, tensor)
    if len(cpu_outputs) != len(qnn_outputs):
        raise RuntimeError("backend output count mismatch")
    return {
        "model": model.name,
        "input_shape": list(tensor.shape),
        "cpu_ms": round(cpu_ms, 3),
        "qnn_ms": round(qnn_ms, 3),
        "outputs": [compare(left, right) for left, right in zip(cpu_outputs, qnn_outputs)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--htp-precision", type=int, choices=(0, 1))
    parser.add_argument("--disable-conv-hmx", action="store_true")
    args = parser.parse_args()
    os.chdir("/app")
    sys.path.insert(0, "/app")
    from ai_edge_litert.interpreter import Interpreter  # noqa: E402
    from utils.tf import load_qnn_delegate  # noqa: E402
    delegate_options = {
        "backend_type": "htp",
        "htp_performance_mode": "2",
        "log_level": "1",
    }
    if args.htp_precision is not None:
        delegate_options["htp_precision"] = str(args.htp_precision)
    if args.disable_conv_hmx:
        delegate_options["htp_use_conv_hmx"] = "false"
    delegate_factory = lambda: load_qnn_delegate(delegate_options=delegate_options)
    print(json.dumps({
        "schema": "vq.mediapipe-tflite-qnn-parity.v1",
        "input": "camera-image" if args.image else "deterministic-linear-minus1-to-plus1",
        "htp_precision": args.htp_precision,
        "htp_use_conv_hmx": not args.disable_conv_hmx,
        "components": [
            run(Interpreter, delegate_factory, args.model_dir / "pose_detector.tflite", args.image),
            run(Interpreter, delegate_factory, args.model_dir / "pose_landmarks_detector.tflite", args.image),
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
