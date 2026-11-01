#!/usr/bin/env python3
"""Display capability 使用示例 — 上层只调 API, 不碰 framebuffer/X11 细节。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import Registry, execute_capability  # noqa: E402
import capability.display as display_pkg  # noqa: E402

reg = Registry()
display_pkg.register(reg)

print("可用的 capability:", reg.list_capabilities())
print("info:", execute_capability(reg, "display.get_info", {}))

# open -> 渲染 -> 截图 -> clear -> close (完整生命周期)
d = execute_capability(reg, "display.open", {"width": 640, "height": 480})
did = d["display_id"]
print(f"\n打开 display {did} ({d['width']}x{d['height']})")

execute_capability(reg, "display.show_text", {"display_id": did, "text": "Hello VQ"})
execute_capability(reg, "display.show_text",
                   {"display_id": did, "text": "Capability Runtime", "position": [20, 60]})
execute_capability(reg, "display.capture",
                   {"display_id": did, "path": "~/vq_display_demo.png"})
print("-> 截图已保存 ~/vq_display_demo.png")

execute_capability(reg, "display.clear", {"display_id": did})
execute_capability(reg, "display.close", {"display_id": did})
print("-> 已 clear 并 close")
