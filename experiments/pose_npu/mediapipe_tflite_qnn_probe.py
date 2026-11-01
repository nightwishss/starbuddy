#!/usr/bin/env python3
"""Bounded QNN/LiteRT load probe for MediaPipe Pose task submodels."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def tensor_for(detail: dict) -> np.ndarray:
    shape = tuple(int(value) for value in detail["shape"])
    dtype = detail["dtype"]
    if np.issubdtype(dtype, np.integer):
        zero_points = detail["quantization_parameters"].get("zero_points", [])
        fill = int(zero_points[0]) if len(zero_points) else 0
        return np.full(shape, fill, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def run_model(interpreter_class, delegate_factory, path: Path, backend: str, iterations: int) -> dict:
    before = time.perf_counter()
    delegates = delegate_factory() if backend == "qnn-htp" else None
    interpreter = interpreter_class(str(path), experimental_delegates=delegates)
    interpreter.allocate_tensors()
    load_ms = (time.perf_counter() - before) * 1000
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1:
        raise RuntimeError(f"expected one input for {path.name}, got {len(inputs)}")
    value = tensor_for(inputs[0])
    interpreter.set_tensor(inputs[0]["index"], value)
    timings = []
    for _ in range(iterations):
        before = time.perf_counter()
        interpreter.invoke()
        timings.append((time.perf_counter() - before) * 1000)
    warm = sorted(timings[1:] or timings)
    return {
        "model": path.name,
        "backend_requested": backend,
        "load_allocate_ms": round(load_ms, 3),
        "input": {
            "shape": [int(item) for item in inputs[0]["shape"]],
            "dtype": np.dtype(inputs[0]["dtype"]).name,
        },
        "outputs": [
            {"shape": [int(item) for item in detail["shape"]],
             "dtype": np.dtype(detail["dtype"]).name}
            for detail in outputs
        ],
        "first_ms": round(timings[0], 3),
        "warm_median_ms": round(statistics.median(warm), 3),
        "warm_p95_ms": round(warm[int((len(warm) - 1) * .95)], 3),
        "warm_min_ms": round(min(warm), 3),
        "warm_max_ms": round(max(warm), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("qnn-htp", "cpu"), required=True)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if not 2 <= args.iterations <= 100:
        parser.error("iterations must be between 2 and 100")
    os.chdir("/app")
    sys.path.insert(0, "/app")
    from ai_edge_litert.interpreter import Interpreter  # noqa: E402
    from utils.tf import load_qnn_delegate  # noqa: E402

    report = {
        "schema": "vq.mediapipe-tflite-backend-smoke.v1",
        "synthetic_input_only": True,
        "backend_requested": args.backend,
        "components": [
            run_model(Interpreter, load_qnn_delegate,
                      args.model_dir / "pose_detector.tflite", args.backend, args.iterations),
            run_model(Interpreter, load_qnn_delegate,
                      args.model_dir / "pose_landmarks_detector.tflite", args.backend, args.iterations),
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
