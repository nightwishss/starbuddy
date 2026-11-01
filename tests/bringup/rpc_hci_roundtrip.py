#!/usr/bin/env python3
"""显式的HCI open -> avail -> close生命周期测试。

此测试会短暂打开Router的hci0通道，因此不属于默认安全回归套件。
无论中途是否失败，finally都会尝试关闭通道。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import RouterRpcClient, RouterRpcError  # noqa: E402


def main():
    client = RouterRpcClient()
    opened = False
    results = []

    def check(cond, label):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    print("=== MPU -> MCU HCI 生命周期测试 ===")
    try:
        client.call("hci/open", ["hci0"])
        opened = True
        check(True, "hci/open成功")

        available = client.call("hci/avail")
        print(f"hci/avail -> {available!r}")
        check(isinstance(available, int), "hci/avail返回整数")

        client.call("hci/close")
        opened = False
        check(True, "hci/close成功")
    except RouterRpcError as exc:
        print(f"  [FAIL] {exc}")
        results.append(False)
    finally:
        if opened:
            try:
                client.call("hci/close")
                print("  [cleanup] HCI通道已关闭")
            except RouterRpcError as exc:
                print(f"  [cleanup FAIL] {exc}")
                results.append(False)

    passed = sum(results)
    print(f"\n=== 结果: {passed}/{len(results)} PASS ===")
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
