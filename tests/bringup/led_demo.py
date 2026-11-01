#!/usr/bin/env python3
"""LED 目视确认 demo — 慢速颜色循环，每色 3 秒，结束自动熄灭。"""
import time
from pathlib import Path

D = Path("/sys/class/leds")
COLORS = ("red", "green", "blue")

def set_all(channel_map):
    """channel_map: {color: 0|255}，其余颜色灭。"""
    for c in COLORS:
        on = channel_map.get(c, 0)
        for p in sorted(D.glob(f"pca963x:{c}*")):
            (p / "brightness").write_text(str(on))

seq = [
    ("红 RED",   {"red": 255}),
    ("绿 GREEN", {"green": 255}),
    ("蓝 BLUE",  {"blue": 255}),
    ("白 WHITE", {"red": 255, "green": 255, "blue": 255}),
]
print("开始慢速颜色循环，每色 3 秒...", flush=True)
for label, m in seq:
    set_all(m)
    print(f"-> {label}", flush=True)
    time.sleep(3)

set_all({})
print("已全部熄灭。", flush=True)
