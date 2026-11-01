#!/usr/bin/env python3
"""LED capability 使用示例 — 上层只调 API, 不碰 /sys 路径。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import Registry, execute_capability  # noqa: E402
import capability.led as led_pkg  # noqa: E402

# 建 registry 并注册 LED 能力
reg = Registry()
led_pkg.register(reg)

print("可用的 capability:", reg.list_capabilities())
print("描述:", reg.describe("led.set_color")["description"])

# 上层视角: 只需要 capability 名 + 参数, 完全看不到 /sys/class/leds
execute_capability(reg, "led.set_color", {"color": "green", "brightness": 128})
print("-> 绿色 128")
execute_capability(reg, "led.set_rgb", {"red": 255, "green": 0, "blue": 0})
print("-> 红色")
execute_capability(reg, "led.off", {})
print("-> 熄灭")
