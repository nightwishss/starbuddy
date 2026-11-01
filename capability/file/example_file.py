#!/usr/bin/env python3
"""File capability 使用示例 — 上层只调 API, 不碰文件系统细节。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import Registry, execute_capability  # noqa: E402
import capability.file as file_pkg  # noqa: E402

reg = Registry()
file_pkg.register(reg)

print("可用的 capability:", reg.list_capabilities())

p = "~/vq_notes/readme.txt"
execute_capability(reg, "file.write", {"path": p, "content": "第一行\n第二行\n"})
print("-> 写了", p)

print("-> 读回:", repr(execute_capability(reg, "file.read", {"path": p})["content"]))
print("-> 目录:", execute_capability(reg, "file.list", {"path": "~/vq_notes"}))
