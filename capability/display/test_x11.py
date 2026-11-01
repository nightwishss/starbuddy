#!/usr/bin/env python3
"""Display X11 backend板上集成测试。

要求指定可用DISPLAY，例如：DISPLAY=:1 python3 capability/display/test_x11.py
测试会创建GTK窗口、更新画面、截图并关闭，最后确认viewer子进程退出。
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import Registry, execute_capability  # noqa: E402
import capability.display as display_pkg  # noqa: E402
from capability.display import impl  # noqa: E402


def main():
    display = os.environ.get("DISPLAY")
    if not display:
        print("FAIL: 必须设置DISPLAY，例如DISPLAY=:1")
        raise SystemExit(2)

    checks = []

    def check(cond, label):
        checks.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    registry = Registry()
    display_pkg.register(registry)
    opened = execute_capability(registry, "display.open", {
        "backend": "x11", "display": display,
        "width": 640, "height": 360, "title": "VQ X11 Review",
    })
    display_id = opened["display_id"]
    backend = impl._get(display_id)
    pid = backend.process.pid

    try:
        check(opened["backend"] == "x11", "open选择X11 backend")
        check(backend.process.poll() is None, f"GTK viewer运行中 pid={pid}")

        execute_capability(registry, "display.clear", {
            "display_id": display_id, "color": "#183153"})
        execute_capability(registry, "display.show_text", {
            "display_id": display_id,
            "text": "VENTUNO Q\nX11 Capability PASS",
            "position": [40, 40],
        })
        time.sleep(0.4)
        check(backend.process.poll() is None, "更新画面后viewer仍存活")

        shot = Path.home() / "vq_x11_review.png"
        execute_capability(registry, "display.capture", {
            "display_id": display_id, "path": str(shot)})
        check(shot.exists() and shot.stat().st_size > 0, "capture生成PNG")
        shot.unlink(missing_ok=True)
    finally:
        execute_capability(registry, "display.close", {"display_id": display_id})

    check(backend.process.poll() is not None, "close终止viewer子进程")
    check(not Path(backend._tmp.name).exists(), "close清理临时目录")
    passed = sum(checks)
    print(f"\n=== 结果: {passed}/{len(checks)} PASS ===")
    raise SystemExit(0 if passed == len(checks) else 1)


if __name__ == "__main__":
    main()
