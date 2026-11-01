#!/usr/bin/env python3
"""Load both pose components on QNN HTP and run bounded synthetic inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep


def create_session(model: Path):
    # Although the ORT API accepts an application-chosen registration key,
    # this plugin reports devices under its canonical EP name. Using that name
    # keeps device selection deterministic across ORT/plugin versions.
    registration_name = qnn_ep.get_ep_names()[0]
    ort.register_execution_provider_library(registration_name, qnn_ep.get_library_path())
    ep_names = qnn_ep.get_ep_names()
    devices = [item for item in ort.get_ep_devices() if item.ep_name in ep_names]
    if not devices:
        raise RuntimeError(f"no QNN EP device found; ep_names={ep_names!r}")
    options = ort.SessionOptions()
    options.log_severity_level = 2
    options.add_provider_for_devices(
        devices,
        {
            "backend_type": "htp",
            "htp_performance_mode": "burst",
        },
    )
    before = time.perf_counter()
    session = ort.InferenceSession(str(model), sess_options=options)
    load_ms = (time.perf_counter() - before) * 1000
    return registration_name, session, load_ms


def run_component(model: Path, iterations: int) -> dict:
    registration_name, session, load_ms = create_session(model)
    try:
        model_input = session.get_inputs()[0]
        shape = tuple(int(item) for item in model_input.shape)
        tensor = np.zeros(shape, dtype=np.float32)
        timings = []
        output_shapes = None
        for _ in range(iterations):
            before = time.perf_counter()
            outputs = session.run(None, {model_input.name: tensor})
            timings.append((time.perf_counter() - before) * 1000)
            output_shapes = [list(item.shape) for item in outputs]
        ordered = sorted(timings)
        return {
            "model": model.name,
            "input": {"name": model_input.name, "shape": list(shape), "dtype": "float32"},
            "output_shapes": output_shapes,
            "session_load_ms": round(load_ms, 3),
            "iterations": iterations,
            "first_inference_ms": round(timings[0], 3),
            "warm_median_ms": round(ordered[len(ordered) // 2], 3),
            "warm_min_ms": round(min(timings), 3),
            "warm_max_ms": round(max(timings), 3),
        }
    finally:
        del session
        ort.unregister_execution_provider_library(registration_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 100:
        parser.error("iterations must be between 1 and 100")

    started = time.perf_counter()
    report = {
        "schema": "vq.pose-npu-smoke.v1",
        "synthetic_input_only": True,
        "backend_requested": "qnn-htp",
        "onnxruntime": ort.__version__,
        "qnn_plugin": getattr(qnn_ep, "__version__", "unknown"),
        "components": [
            run_component(args.model_dir / "pose_detector.onnx", args.iterations),
            run_component(args.model_dir / "pose_landmark_detector.onnx", args.iterations),
        ],
    }
    report["wall_ms"] = round((time.perf_counter() - started) * 1000, 3)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
