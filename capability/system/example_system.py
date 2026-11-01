#!/usr/bin/env python3
"""System capability 使用示例 — 上层只调 API, 不碰 /proc 路径。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import Registry, execute_capability  # noqa: E402
import capability.system as sys_pkg  # noqa: E402

reg = Registry()
sys_pkg.register(reg)

print("可用的 capability:", reg.list_capabilities())
print("描述:", reg.describe("system.status")["description"])

info = execute_capability(reg, "system.info", {})
print(f"\n系统: {info['hostname']} ({info['machine']}) 内核 {info['kernel']}")

st = execute_capability(reg, "system.status", {})
print(f"负载: {st['loadavg']}  CPU: {st['cpu_percent']}%  "
      f"内存: {st['mem']['percent']}%  温度: {st['temp_max_c']}°C")
