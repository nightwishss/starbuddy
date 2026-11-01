#!/usr/bin/env python3
"""Display Capability 测试 — Resource/Stateful 生命周期 + 渲染 + 错误模型.

测试 A-E: 完整 open -> show_text -> show_image -> clear -> capture -> close 生命周期
测试 F:   错误模型 (invalid display_id / invalid resolution / missing param / 非法图片)

运行 (项目根目录):  python3 capability/display/test_display.py
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import (  # noqa: E402
    Registry, execute_capability,
    HardwareError, SchemaValidationError, ResourceNotFoundError,
)
import capability.display as display_pkg  # noqa: E402


def main():
    results = []

    def check(cond, label):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    print("=== Display Capability 测试 (Resource/Stateful) ===")
    reg = Registry()
    display_pkg.register(reg)
    names = reg.list_capabilities()
    print(f"\n注册的 capability ({len(names)}): {names}")
    check(len(names) == 8, "8 个 display capability")

    # 类型检查
    check(reg.describe("display.open")["type"] == "stateful", "display.open type == stateful")
    check(reg.describe("display.close")["type"] == "stateful", "display.close type == stateful")
    check(reg.describe("display.list")["type"] == "query", "display.list type == query")
    check(reg.describe("display.show_text")["type"] == "command", "display.show_text type == command")

    print("\n--- A. display.get_info / list ---")
    info = execute_capability(reg, "display.get_info", {})
    print("  info:", info)
    check(info["backend"] == "framebuffer" and info["physical"] is False,
          "get_info: software framebuffer backend")
    lst = execute_capability(reg, "display.list", {})
    check(lst["count"] == 0, "初始无活跃 display 会话")

    print("\n--- A. display.open (创建 virtual display) ---")
    opened = execute_capability(reg, "display.open", {"width": 640, "height": 480})
    did = opened["display_id"]
    print(f"  display_id={did}, {opened['width']}x{opened['height']}")
    check(bool(did) and len(did) == 8, f"open 返回 8 位 display_id ({did})")
    lst = execute_capability(reg, "display.list", {})
    check(lst["count"] == 1 and did in lst["active"], "list 显示 1 个活跃会话")

    print("\n--- B. display.show_text ---")
    execute_capability(reg, "display.show_text", {"display_id": did, "text": "Hello VQ"})
    execute_capability(reg, "display.show_text",
                       {"display_id": did, "text": "Capability Runtime", "position": [20, 60]})
    print("  渲染了 'Hello VQ' + 'Capability Runtime'")

    print("\n--- C. display.show_image (生成测试图并显示) ---")
    from PIL import Image, ImageDraw
    img_path = Path.home() / "vq_test_pattern.png"
    im = Image.new("RGB", (200, 150), (30, 60, 120))
    d = ImageDraw.Draw(im)
    d.rectangle((30, 30, 170, 120), fill=(220, 120, 40))
    im.save(img_path)
    r = execute_capability(reg, "display.show_image",
                           {"display_id": did, "path": str(img_path)})
    check(r["rendered"] is True, "show_image 渲染测试图")

    print("\n--- screenshot (capture) ---")
    shot = Path.home() / "vq_display_test.png"
    r = execute_capability(reg, "display.capture",
                           {"display_id": did, "path": str(shot)})
    check(shot.exists() and shot.stat().st_size > 0,
          f"capture 生成截图 {shot} ({shot.stat().st_size} bytes)")

    print("\n--- D. display.clear ---")
    r = execute_capability(reg, "display.clear", {"display_id": did})
    check(r["cleared"] is True, "clear 成功")

    print("\n--- E. display.close ---")
    r = execute_capability(reg, "display.close", {"display_id": did})
    check(r["closed"] is True, "close 成功")
    lst = execute_capability(reg, "display.list", {})
    check(lst["count"] == 0, "close 后无活跃会话")

    print("\n--- F. 错误模型 ---")
    # F1: invalid display_id (close 后再操作)
    try:
        execute_capability(reg, "display.show_text",
                           {"display_id": did, "text": "x"})
        check(False, "close 后再操作应报错")
    except ResourceNotFoundError as e:
        check(e.code == "RESOURCE_NOT_FOUND", "close 后再操作 -> RESOURCE_NOT_FOUND")
    # F2: 不存在的 display_id
    try:
        execute_capability(reg, "display.show_text",
                           {"display_id": "deadbeef", "text": "x"})
        check(False, "不存在 display_id 应报错")
    except ResourceNotFoundError:
        check(True, "不存在 display_id -> RESOURCE_NOT_FOUND")
    # F3: invalid resolution (低于最小值)
    try:
        execute_capability(reg, "display.open", {"width": 10, "height": 10})
        check(False, "width=10 应报 schema 错误")
    except SchemaValidationError:
        check(True, "invalid resolution -> SCHEMA_VALIDATION_ERROR")
    # F4: missing parameter (show_text 缺 text)
    try:
        d2 = execute_capability(reg, "display.open", {})
        execute_capability(reg, "display.show_text", {"display_id": d2["display_id"]})
        check(False, "缺 text 应报 schema 错误")
    except SchemaValidationError:
        check(True, "missing parameter -> SCHEMA_VALIDATION_ERROR")
    finally:
        execute_capability(reg, "display.close", {"display_id": d2["display_id"]})
    # F5: 非法图片格式 (文本文件当图片)
    try:
        d3 = execute_capability(reg, "display.open", {})
        txt = Path.home() / "vq_bad.txt"
        txt.write_text("not an image")
        execute_capability(reg, "display.show_image",
                           {"display_id": d3["display_id"], "path": str(txt)})
        check(False, "非图片文件应报错")
    except SchemaValidationError:
        check(True, "unsupported image format -> SCHEMA_VALIDATION_ERROR")
    finally:
        execute_capability(reg, "display.close", {"display_id": d3["display_id"]})

    # F6: X11 backend无DISPLAY时必须明确失败，且不能注册半成品资源
    old_display = os.environ.pop("DISPLAY", None)
    try:
        execute_capability(reg, "display.open", {"backend": "x11"})
        check(False, "X11无DISPLAY应报硬件错误")
    except HardwareError:
        check(True, "X11无DISPLAY -> HARDWARE_ERROR")
    finally:
        if old_display is not None:
            os.environ["DISPLAY"] = old_display
    check(execute_capability(reg, "display.list", {})["count"] == 0,
          "X11启动失败不残留display资源")

    # 清理测试产物
    for p in (img_path, shot, Path.home() / "vq_bad.txt"):
        try:
            p.unlink()
        except OSError:
            pass

    total = len(results)
    passed = sum(results)
    print(f"\n=== 结果: {passed}/{total} PASS ===")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
