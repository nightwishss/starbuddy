#!/usr/bin/env python3
"""System capability 测试 — 纯只读查询 + 错误模型测试。

运行 (项目根目录):  python3 capability/system/test_system.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import Registry, execute_capability  # noqa: E402
import capability.system as sys_pkg  # noqa: E402


def main():
    results = []

    def check(cond, label):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    print("=== System Capability 测试 (Query 型) ===")

    reg = Registry()
    sys_pkg.register(reg)
    names = reg.list_capabilities()
    print(f"\n注册的 capability: {names}")
    check(sorted(names) == ["system.info", "system.status"],
          "list_capabilities 返回 2 个 system capability")

    check(reg.describe("system.status")["type"] == "query",
          "system.status 的 type == query")

    print("\n--- system.info ---")
    info = execute_capability(reg, "system.info", {})
    print("  ", info)
    check(info["hostname"] and info["kernel"], "system.info 返回 hostname/kernel")

    print("\n--- system.status ---")
    st = execute_capability(reg, "system.status", {})
    for k in ("loadavg", "cpu_percent", "cpu_count", "mem", "disk",
              "uptime_seconds", "boot_time", "temp_max_c"):
        print(f"  {k}: {st[k]}")
    check(len(st["loadavg"]) == 3, "loadavg 3 项 (1/5/15 分钟)")
    check(isinstance(st["cpu_count"], int) and st["cpu_count"] >= 8,
          f"cpu_count = {st['cpu_count']} (期望 >= 8, 实际 8 核)")
    check(st["mem"]["total_kb"] > 0 and st["mem"]["available_kb"] > 0,
          "内存 total/available > 0")
    check(st["disk"]["total_gb"] > 0, f"磁盘 total = {st['disk']['total_gb']} GB")
    check(st["uptime_seconds"] > 0, f"uptime = {st['uptime_seconds']} s")
    check(st["temp_max_c"] >= 0, f"temp_max = {st['temp_max_c']} °C")

    total = len(results)
    passed = sum(results)
    print(f"\n=== 结果: {passed}/{total} PASS ===")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
