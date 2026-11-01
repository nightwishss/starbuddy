#!/usr/bin/env python3
"""GPIO capability 测试 (MPU 只读) — 枚举 chip/line + 读电平。

运行 (项目根目录):  python3 capability/gpio/test_gpio.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import (  # noqa: E402
    Registry, execute_capability,
    CapabilityNotFoundError, HardwareError,
)
import capability.gpio as gpio_pkg  # noqa: E402


def main():
    results = []

    def check(cond, label):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    print("=== GPIO Capability 测试 (MPU 只读) ===")

    reg = Registry()
    gpio_pkg.register(reg)
    names = reg.list_capabilities()
    print(f"\n注册的 capability: {names}")
    check(sorted(names) == ["gpio.list_chips", "gpio.list_lines", "gpio.read"],
          "list_capabilities 返回 3 个 gpio capability")

    print("\n--- gpio.list_chips ---")
    r = execute_capability(reg, "gpio.list_chips", {})
    for c in r["chips"]:
        print(f"  {c['name']} [{c['label']}]: {c['num_lines']} lines")
    check(r["count"] == 4, f"4 个 gpiochip (实际 {r['count']})")
    total = sum(c["num_lines"] for c in r["chips"])
    check(total == 181, f"总 line = {total} (期望 181)")

    print("\n--- gpio.list_lines (gpiochip2) ---")
    r = execute_capability(reg, "gpio.list_lines", {"chip": "gpiochip2"})
    check(r["count"] == 134, f"gpiochip2 = {r['count']} lines (期望 134)")

    print("\n--- gpio.read (读 unused input line) ---")
    # 从 gpioinfo 找一条 unused input line, 用 gpioget 读 (bring-up 已验证 gpiochip1/2/3 可读)
    read_ok = False
    for chip in ("gpiochip3", "gpiochip2", "gpiochip1"):
        lines = execute_capability(reg, "gpio.list_lines", {"chip": chip})["lines"]
        cand = None
        for ln in lines:
            if "input" in ln and "[used]" not in ln:
                off = int(ln.split("line")[1].split(":")[0].strip())
                cand = off
                break
        if cand is None:
            continue
        try:
            v = execute_capability(reg, "gpio.read", {"chip": chip, "offset": cand})
            print(f"  {chip} line {cand}: value={v['value']} (读通路 OK)")
            read_ok = True
            break
        except HardwareError as e:
            print(f"  {chip} line {cand}: 读失败 -> {e.message}")
    check(read_ok, "gpioget 读通路正常")

    print("\n--- 错误模型: 不存在的 chip ---")
    try:
        execute_capability(reg, "gpio.list_lines", {"chip": "gpiochip99"})
        check(False, "不存在 chip 应报错")
    except HardwareError:
        check(True, "不存在 chip -> HARDWARE_ERROR")

    total = len(results)
    passed = sum(results)
    print(f"\n=== 结果: {passed}/{total} PASS ===")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
