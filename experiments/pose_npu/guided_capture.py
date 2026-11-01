#!/usr/bin/env python3
"""Capture a spoken, labelled posture sequence only in tmpfs."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2


SEGMENTS = (
    ("healthy_baseline", 15, "第一段，请自然坐直，保持健康坐姿十五秒。"),
    ("head_forward", 20, "第二段，请低头并把头向前伸，保持二十秒。"),
    ("slouch_forward", 20, "第三段，请含胸并让上半身向前，保持二十秒。"),
    ("shoulder_lean", 20, "第四段，请让一侧肩膀明显低一些，保持二十秒。"),
    ("healthy_recovery", 10, "最后一段，请恢复自然坐直，保持十秒。"),
)


def announce(root: Path, text: str):
    subprocess.run(["python3", str(root / "tools/pose_npu_announce.py"), text],
                   cwd=root, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-url", default="http://127.0.0.1:8765/api/camera/live.mjpg")
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--rotation", type=int, choices=(0, 180), default=180)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    resolved = args.output_dir.resolve()
    if not str(resolved).startswith("/dev/shm/vq-pose-shadow-"):
        parser.error("output-dir must be a dedicated /dev/shm/vq-pose-shadow-* directory")
    if not 1 <= args.fps <= 5:
        parser.error("fps must be between 1 and 5")
    resolved.mkdir(mode=0o700, parents=False, exist_ok=False)

    capture = cv2.VideoCapture(args.source_url)
    if not capture.isOpened():
        raise SystemExit("camera stream unavailable")
    frames, frame_labels, elapsed = [], {}, {}
    test_started = time.monotonic()
    try:
        announce(args.repo_root, "测试即将开始。听到每段提示后，请保持对应姿势。")
        for label, duration, cue in SEGMENTS:
            announce(args.repo_root, cue)
            deadline = time.monotonic() + duration
            next_frame = time.monotonic()
            while time.monotonic() < deadline:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("camera stream disconnected")
                now = time.monotonic()
                if now < next_frame:
                    continue
                next_frame = now + 1.0 / args.fps
                if args.rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                name = f"frame-{len(frames):04d}.jpg"
                if not cv2.imwrite(str(resolved / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                    raise RuntimeError(f"failed to write {name}")
                frames.append(name)
                frame_labels[name] = label
                elapsed[name] = round(now - test_started, 3)
        announce(args.repo_root, "姿态动作采集完成，谢谢。现在开始在板端对比两个模型。")
    finally:
        capture.release()
    manifest = {
        "schema": "vq.pose-shadow-frames.v2",
        "label": "guided-posture-state-agreement",
        "frames": frames,
        "frame_count": len(frames),
        "frame_labels": frame_labels,
        "frame_elapsed_seconds": elapsed,
        "segments": [{"label": label, "duration_seconds": duration}
                     for label, duration, _ in SEGMENTS],
        "fps_target": args.fps,
        "rotation_applied": args.rotation,
        "storage": "tmpfs",
        "delete_after_comparison": True,
    }
    (resolved / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: value for key, value in manifest.items()
                      if key not in {"frames", "frame_labels", "frame_elapsed_seconds"}}, indent=2))


if __name__ == "__main__":
    main()
