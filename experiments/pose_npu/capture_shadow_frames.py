#!/usr/bin/env python3
"""Capture a bounded, rotated frame set for same-frame CPU/NPU comparison."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-url", default="http://127.0.0.1:8765/api/camera/live.mjpg")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--rotation", type=int, choices=(0, 180), default=180)
    args = parser.parse_args()
    if not 1 <= args.duration <= 60:
        parser.error("duration must be between 1 and 60 seconds")
    if not 1 <= args.fps <= 10:
        parser.error("fps must be between 1 and 10")
    resolved = args.output_dir.resolve()
    if not str(resolved).startswith("/dev/shm/vq-pose-shadow-"):
        parser.error("output-dir must be a dedicated /dev/shm/vq-pose-shadow-* directory")
    resolved.mkdir(mode=0o700, parents=False, exist_ok=False)

    capture = cv2.VideoCapture(args.source_url)
    if not capture.isOpened():
        raise SystemExit("camera stream unavailable")
    deadline = time.monotonic() + args.duration
    interval = 1.0 / args.fps
    next_frame = time.monotonic()
    frames = []
    try:
        while time.monotonic() < deadline:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("camera stream disconnected")
            now = time.monotonic()
            if now < next_frame:
                continue
            next_frame = now + interval
            if args.rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            name = f"frame-{len(frames):04d}.jpg"
            if not cv2.imwrite(str(resolved / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"failed to write {name}")
            frames.append(name)
    finally:
        capture.release()
    manifest = {
        "schema": "vq.pose-shadow-frames.v1",
        "label": args.label,
        "frames": frames,
        "frame_count": len(frames),
        "fps_target": args.fps,
        "duration_target_seconds": args.duration,
        "rotation_applied": args.rotation,
        "storage": "tmpfs",
        "delete_after_comparison": True,
    }
    (resolved / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
