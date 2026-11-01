#!/usr/bin/env python3
"""Arduino Router 安全冒烟测试。

只调用Router自身的只读方法，不打开HCI、不发送数据、不改变MCU或蓝牙状态。
需要验证MCU/HCI生命周期时，显式运行rpc_hci_roundtrip.py。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from runtime import RouterRpcClient, RouterRpcError  # noqa: E402


def main():
    client = RouterRpcClient()
    results = []

    def check(cond, label):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    print("=== Arduino Router 安全 RPC 冒烟测试 ===")
    try:
        version = client.call("$/version")
        print(f"$/version -> {version!r}")
        check(isinstance(version, str) and bool(version), "Router返回版本号")

        connected = client.call("mon/connected")
        print(f"mon/connected -> {connected!r}")
        check(isinstance(connected, bool), "monitor连接状态为boolean")
    except RouterRpcError as exc:
        print(f"  [FAIL] {exc}")
        results.append(False)

    passed = sum(results)
    print(f"\n=== 结果: {passed}/{len(results)} PASS ===")
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
