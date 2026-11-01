#!/usr/bin/env python3
"""File capability 测试 — 读写/列目录 + 路径穿越防护。

运行 (项目根目录):  python3 capability/file/test_file.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import (  # noqa: E402
    Registry, execute_capability,
    SchemaValidationError, HardwareError,
)
import capability.file as file_pkg  # noqa: E402


def main():
    results = []

    def check(cond, label):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    print("=== File Capability 测试 (query/command + 路径防护) ===")

    reg = Registry()
    file_pkg.register(reg)
    names = reg.list_capabilities()
    print(f"\n注册的 capability: {names}")
    check(sorted(names) == ["file.exists", "file.list", "file.read", "file.write"],
          "list_capabilities 返回 4 个 file capability")

    # 用临时目录 (在 ~ 内) 做读写, 避免污染
    base = Path.home() / ".vq_test_file"
    base.mkdir(parents=True, exist_ok=True)
    f = base / "hello.txt"

    print("\n--- file.write ---")
    r = execute_capability(reg, "file.write",
                           {"path": str(f), "content": "hello ventunoq\n"})
    check(r["bytes"] > 0, f"write 返回 bytes={r['bytes']}")

    print("\n--- file.read ---")
    r = execute_capability(reg, "file.read", {"path": str(f)})
    check(r["content"] == "hello ventunoq\n", f"read 读回一致: {r['content']!r}")

    print("\n--- file.exists ---")
    check(execute_capability(reg, "file.exists", {"path": str(f)})["exists"],
          "exists(已写文件) == true")
    check(not execute_capability(reg, "file.exists",
          {"path": str(base / "nope.txt")})["exists"], "exists(不存在) == false")

    print("\n--- file.list ---")
    r = execute_capability(reg, "file.list", {"path": str(base)})
    check(r["count"] >= 1 and any(e["name"] == "hello.txt" for e in r["entries"]),
          "list 包含刚写的 hello.txt")

    print("\n--- 路径穿越防护 ---")
    try:
        execute_capability(reg, "file.read", {"path": "/etc/passwd"})
        check(False, "读 /etc/passwd 应被拒绝 (越界)")
    except SchemaValidationError as e:
        check(e.code == "SCHEMA_VALIDATION_ERROR", "越界路径 -> SCHEMA_VALIDATION_ERROR")
    # 相对越界 (从 home 向上逃出, 与 cwd 无关)
    outside = str(Path.home().parent / "etc" / "passwd")  # /home/etc/passwd, 逃出 ~
    try:
        execute_capability(reg, "file.read", {"path": outside})
        check(False, f"读 {outside} 应被拒绝 (越界)")
    except SchemaValidationError:
        check(True, "相对越界路径 -> SCHEMA_VALIDATION_ERROR")

    # 清理测试文件
    import shutil
    shutil.rmtree(base, ignore_errors=True)
    print(f"\n(已清理测试目录 {base})")

    total = len(results)
    passed = sum(results)
    print(f"\n=== 结果: {passed}/{total} PASS ===")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
