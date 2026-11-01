"""Arduino Router 的最小 msgpack-RPC 客户端。"""

from __future__ import annotations

import socket
from itertools import count

import msgpack


class RouterRpcError(RuntimeError):
    """Router RPC 客户端错误基类。"""


class RouterRpcProtocolError(RouterRpcError):
    """Router 返回了不符合 msgpack-RPC 约定的数据。"""


class RouterRpcRemoteError(RouterRpcError):
    """Router 或 MCU 返回了结构化远端错误。"""

    def __init__(self, method, error):
        self.method = method
        self.error = error
        super().__init__(f"{method} RPC 失败: {error!r}")


class RouterRpcClient:
    """一次调用建立一次 Unix socket 连接的轻量客户端。"""

    def __init__(self, socket_path="/var/run/arduino-router.sock", timeout=6.0):
        self.socket_path = socket_path
        self.timeout = timeout
        self._ids = count(1)

    def call(self, method, params=None):
        msgid = next(self._ids)
        request = [0, msgid, method, [] if params is None else params]
        unpacker = msgpack.Unpacker(raw=False)

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout)
                sock.connect(self.socket_path)
                sock.sendall(msgpack.packb(request, use_bin_type=True))
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        raise RouterRpcProtocolError(
                            f"{method} 连接关闭但未收到完整响应")
                    unpacker.feed(chunk)
                    for response in unpacker:
                        return self._parse_response(method, msgid, response)
        except socket.timeout as exc:
            raise RouterRpcError(f"{method} RPC 超时 ({self.timeout}s)") from exc
        except OSError as exc:
            raise RouterRpcError(
                f"无法访问 Arduino Router {self.socket_path}: {exc}") from exc

    @staticmethod
    def _parse_response(method, msgid, response):
        if not isinstance(response, (list, tuple)) or len(response) != 4:
            raise RouterRpcProtocolError(f"{method} 非法响应: {response!r}")
        msg_type, response_id, error, result = response
        if msg_type != 1:
            raise RouterRpcProtocolError(
                f"{method} 响应类型应为 1，实际为 {msg_type!r}")
        if response_id != msgid:
            raise RouterRpcProtocolError(
                f"{method} 响应 ID {response_id!r} != 请求 ID {msgid!r}")
        if error not in (None, "", b""):
            raise RouterRpcRemoteError(method, error)
        return result

