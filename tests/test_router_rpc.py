#!/usr/bin/env python3
"""RouterRpcClient 的无硬件单元测试。"""

import socket
import sys
import tempfile
import threading
from pathlib import Path

import msgpack

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime import (  # noqa: E402
    RouterRpcClient,
    RouterRpcProtocolError,
    RouterRpcRemoteError,
)


def serve_once(path, build_response):
    ready = threading.Event()

    def server():
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.bind(path)
            sock.listen(1)
            ready.set()
            conn, _ = sock.accept()
            with conn:
                request = msgpack.unpackb(conn.recv(4096), raw=False)
                response = build_response(request)
                conn.sendall(msgpack.packb(response, use_bin_type=True))

    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    ready.wait(2)
    return thread


def main():
    checks = []

    def check(cond, label):
        checks.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "router.sock")
        thread = serve_once(path, lambda req: [1, req[1], None, "0.10.0"])
        result = RouterRpcClient(path).call("$/version")
        thread.join(2)
        check(result == "0.10.0", "解析成功响应")

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "router.sock")
        thread = serve_once(path, lambda req: [1, req[1], [2, "remote failure"], None])
        try:
            RouterRpcClient(path).call("hci/avail")
            check(False, "远端错误应抛异常")
        except RouterRpcRemoteError as exc:
            check(exc.method == "hci/avail", "远端错误保留method")
        thread.join(2)

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "router.sock")
        thread = serve_once(path, lambda req: [1, req[1] + 1, None, True])
        try:
            RouterRpcClient(path).call("mon/connected")
            check(False, "错误msgid应抛协议异常")
        except RouterRpcProtocolError:
            check(True, "拒绝错误msgid")
        thread.join(2)

    passed = sum(checks)
    print(f"\n=== 结果: {passed}/{len(checks)} PASS ===")
    raise SystemExit(0 if passed == len(checks) else 1)


if __name__ == "__main__":
    main()
