#!/usr/bin/env python3
"""GPIO capability 使用示例 — 上层只调 API, 不碰 libgpiod CLI。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import Registry, execute_capability  # noqa: E402
import capability.gpio as gpio_pkg  # noqa: E402

reg = Registry()
gpio_pkg.register(reg)

print("可用的 capability:", reg.list_capabilities())
print("描述:", reg.describe("gpio.read")["description"])

chips = execute_capability(reg, "gpio.list_chips", {})
print(f"\n共 {chips['count']} 个 gpiochip:")
for c in chips["chips"]:
    print(f"  {c['name']}  {c['label']}  ({c['num_lines']} lines)")
