#!/usr/bin/env python3
"""LED capability 测试 — 含实际硬件测试 + 错误模型测试。

运行 (项目根目录):  python3 capability/led/test_led.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import (  # noqa: E402
    Registry, execute_capability,
    SchemaValidationError, CapabilityNotFoundError,
)
import capability.led as led_pkg  # noqa: E402


def main():
    results = []

    def check(cond, label):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    print("=== LED Capability 测试 (Command/Query 接口 + 错误模型) ===")

    # 1. 注册
    reg = Registry()
    led_pkg.register(reg)
    names = reg.list_capabilities()
    print(f"\n注册的 capability: {names}")
    check(sorted(names) == ["led.list", "led.off", "led.set_color", "led.set_rgb"],
          "list_capabilities 返回 4 个 LED capability")

    # 2. describe / type 字段
    check(reg.describe("led.set_color")["type"] == "command",
          "led.set_color 的 type == command")
    check(reg.describe("led.list")["type"] == "query",
          "led.list 的 type == query (Query 型示例)")

    # 3. 错误模型: capability 不存在
    try:
        execute_capability(reg, "led.nope", {})
        check(False, "不存在的 capability 应报错")
    except CapabilityNotFoundError as e:
        check(e.code == "CAPABILITY_NOT_FOUND", "CAPABILITY_NOT_FOUND")

    # 4. 错误模型: 非法 color (enum 校验)
    try:
        execute_capability(reg, "led.set_color",
                           {"color": "purple", "brightness": 255})
        check(False, "非法 color 应报 schema 错误")
    except SchemaValidationError as e:
        check(e.code == "SCHEMA_VALIDATION_ERROR", "SCHEMA_VALIDATION_ERROR (非法 color)")

    # 5. 错误模型: 缺少必填字段
    try:
        execute_capability(reg, "led.set_color", {"color": "red"})
        check(False, "缺少 brightness 应报 schema 错误")
    except SchemaValidationError:
        check(True, "缺少必填字段 -> SCHEMA_VALIDATION_ERROR")

    # 6. 错误模型: 越界值 (maximum)
    try:
        execute_capability(reg, "led.set_color",
                           {"color": "red", "brightness": 999})
        check(False, "brightness=999 应报 schema 错误")
    except SchemaValidationError:
        check(True, "brightness 越界 -> SCHEMA_VALIDATION_ERROR")

    # 7. 实际硬件: set_color 红 200, 读回校验
    print("\n=== 实际硬件测试 (写入 sysfs 并读回校验) ===")
    r = execute_capability(reg, "led.set_color",
                           {"color": "red", "brightness": 200})
    check(r["count"] == 4, f"set_color red=200 影响 {r['count']} 通道")
    lst = execute_capability(reg, "led.list", {})
    reds = [b for ch, b in lst["brightness"].items()
            if ch.startswith("pca963x:red")]
    check(all(b == 200 for b in reds), f"读回红色通道亮度 = {reds}")

    # 8. 实际硬件: set_rgb 写 12 通道
    r = execute_capability(reg, "led.set_rgb",
                           {"red": 0, "green": 255, "blue": 0})
    check(r["count"] == 12, f"set_rgb 写 {r['count']} 通道 (期望 12)")

    # 9. 物理颜色循环 (供目视)
    print("\n颜色循环 (每色 1.5s):")
    for label, rgb in [("红", (255, 0, 0)), ("绿", (0, 255, 0)),
                       ("蓝", (0, 0, 255)), ("白", (255, 255, 255))]:
        execute_capability(reg, "led.set_rgb",
                           {"red": rgb[0], "green": rgb[1], "blue": rgb[2]})
        print(f"  -> {label} {rgb}")
        time.sleep(1.5)

    # 10. off + 读回校验
    r = execute_capability(reg, "led.off", {})
    lst = execute_capability(reg, "led.list", {})
    all_off = all(b == 0 for b in lst["brightness"].values())
    check(r["all_off"] and all_off, "led.off 全部熄灭并读回为 0")

    total = len(results)
    passed = sum(results)
    print(f"\n=== 结果: {passed}/{total} PASS ===")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
