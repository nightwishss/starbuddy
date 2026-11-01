#!/usr/bin/env python3
"""Audio Capability 测试 — Stream 模型 + File backend + 错误模型.

测试:
  A. 生成 1kHz sine WAV (2s, 16kHz, mono)
  B. open_playback 读流信息 + play 分块读取 (验证 stream 生命周期)
  C/D. 音量处理 (×0.5) -> open_capture + write -> close_capture 输出新 WAV
  E. 重新读取 output WAV, 验证 metadata + 波形峰值约减半
  F. 错误模型: invalid stream_id / missing param / 不存在文件

运行 (项目根目录):  python3 capability/audio/test_audio.py
"""
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import (  # noqa: E402
    Registry, execute_capability,
    ResourceNotFoundError, SchemaValidationError, HardwareError,
)
import capability.audio as audio_pkg  # noqa: E402


def main():
    results = []

    def check(cond, label):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    print("=== Audio Capability 测试 (Stream 模型 + File backend) ===")
    reg = Registry()
    audio_pkg.register(reg)
    names = reg.list_capabilities()
    print(f"\n注册的 capability ({len(names)}): {names}")
    check(len(names) == 8, "8 个 audio capability")

    # 类型检查
    check(reg.describe("audio.open_playback")["type"] == "stateful", "open_playback == stateful")
    check(reg.describe("audio.open_capture")["type"] == "stateful", "open_capture == stateful")
    check(reg.describe("audio.play")["type"] == "command", "play == command")
    check(reg.describe("audio.list_devices")["type"] == "query", "list_devices == query")

    print("\n--- list_devices / get_info ---")
    devs = execute_capability(reg, "audio.list_devices", {})
    print("  devices:", devs)
    check(devs["count"] >= 1 and "file" in devs["backends"], "list_devices 返回 backend + 声卡")
    info = execute_capability(reg, "audio.get_info", {})
    print("  info:", info)
    check(info["backend"] == "file", "get_info backend == file")

    print("\n--- A. 生成 1kHz sine WAV (16kHz, 2s, mono) ---")
    sr, dur = 16000, 2.0
    n = int(sr * dur)
    t = np.arange(n) / sr
    src = (0.5 * 32767 * np.sin(2 * np.pi * 1000 * t)).astype(np.int16)
    src_path = Path.home() / "vq_sine.wav"
    wf = wave.open(str(src_path), "wb")
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
    wf.writeframes(src.tobytes())
    wf.close()
    check(src_path.exists() and src_path.stat().st_size > 0, f"生成源 WAV ({n} frames)")

    print("\n--- B. open_playback + play 分块读 ---")
    sp = execute_capability(reg, "audio.open_playback", {"path": str(src_path)})
    sid = sp["stream_id"]
    print(f"  stream_id={sid}, {sp['sample_rate']}Hz {sp['channels']}ch "
          f"{sp['frames']}frames {sp['duration_seconds']}s")
    check(sp["sample_rate"] == 16000, "sample_rate == 16000")
    check(sp["channels"] == 1 and sp["sample_width"] == 2, "mono 16-bit")
    check(sp["frames"] == n, f"frames == {n}")
    check(abs(sp["duration_seconds"] - 2.0) < 0.01, "duration ≈ 2.0s")

    total = 0
    all_samples = []
    eof = False
    while not eof:
        r = execute_capability(reg, "audio.play", {"stream_id": sid, "frames": 1024})
        total += r["frames_read"]
        all_samples.extend(r["samples"])
        eof = r["eof"]
    check(total == n, f"分块读总帧数 == {n}")
    check(len(all_samples) == n, f"累计样本数 == {n}")
    peak_in = int(max(abs(s) for s in all_samples))
    check(peak_in > 10000, f"正弦峰值 {peak_in} > 10000 (数据非零)")
    execute_capability(reg, "audio.close_playback", {"stream_id": sid})

    print("\n--- C/D. 音量处理 ×0.5 -> capture + write ---")
    processed = [int(s * 0.5) for s in all_samples]
    out_path = Path.home() / "vq_out.wav"
    cp = execute_capability(reg, "audio.open_capture", {
        "path": str(out_path), "sample_rate": 16000, "channels": 1, "sample_width": 2})
    cid = cp["stream_id"]
    for i in range(0, len(processed), 1024):
        execute_capability(reg, "audio.write",
                           {"stream_id": cid, "samples": processed[i:i + 1024]})
    rc = execute_capability(reg, "audio.close_capture", {"stream_id": cid})
    check(rc["closed"] and out_path.exists(), "capture close + 输出 WAV 存在")

    print("\n--- E. 重新读取 output WAV 验证 ---")
    sp2 = execute_capability(reg, "audio.open_playback", {"path": str(out_path)})
    check(sp2["sample_rate"] == 16000 and sp2["channels"] == 1, "output 采样率/声道一致")
    check(sp2["frames"] == n, f"output frames == {n}")
    r = execute_capability(reg, "audio.play", {"stream_id": sp2["stream_id"], "frames": n})
    out_samples = r["samples"]
    check(len(out_samples) == n, f"output 样本数 == {n}")
    peak_out = max(abs(s) for s in out_samples)
    check(abs(peak_out - peak_in * 0.5) < peak_in * 0.05,
          f"峰值约减半: {peak_in} -> {peak_out}")
    execute_capability(reg, "audio.close_playback", {"stream_id": sp2["stream_id"]})

    print("\n--- F. 错误模型 ---")
    try:
        execute_capability(reg, "audio.play", {"stream_id": "deadbeef", "frames": 10})
        check(False, "不存在 stream_id 应报错")
    except ResourceNotFoundError as e:
        check(e.code == "RESOURCE_NOT_FOUND", "invalid stream_id -> RESOURCE_NOT_FOUND")
    try:
        execute_capability(reg, "audio.play", {"frames": 10})
        check(False, "缺 stream_id 应报 schema 错误")
    except SchemaValidationError:
        check(True, "missing parameter -> SCHEMA_VALIDATION_ERROR")
    try:
        execute_capability(reg, "audio.open_playback", {"path": "/nonexistent.wav"})
        check(False, "不存在文件应报错")
    except HardwareError:
        check(True, "文件不存在 -> HARDWARE_ERROR")

    # 清理
    for p in (src_path, out_path):
        try:
            p.unlink()
        except OSError:
            pass

    total_r = len(results)
    passed = sum(results)
    print(f"\n=== 结果: {passed}/{total_r} PASS ===")
    sys.exit(0 if passed == total_r else 1)


if __name__ == "__main__":
    main()
