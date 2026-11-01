#!/usr/bin/env python3
"""Audio capability 使用示例 — 上层只调 API, 不碰 WAV/ALSA 细节。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import Registry, execute_capability  # noqa: E402
import capability.audio as audio_pkg  # noqa: E402

reg = Registry()
audio_pkg.register(reg)

print("可用的 capability:", reg.list_capabilities())
print("info:", execute_capability(reg, "audio.get_info", {}))

# 打开 playback 流读一段 PCM
sp = execute_capability(reg, "audio.open_playback", {"path": "~/vq_sine.wav"})
print(f"\n流: {sp['sample_rate']}Hz {sp['channels']}ch {sp['frames']} frames")
r = execute_capability(reg, "audio.play", {"stream_id": sp["stream_id"], "frames": 1000})
print(f"读到 {r['frames_read']} frames, eof={r['eof']}")
execute_capability(reg, "audio.close_playback", {"stream_id": sp["stream_id"]})
