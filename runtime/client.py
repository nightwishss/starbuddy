"""VQ Runtime Unix Socket JSON客户端。"""

import json
import socket
import uuid

from runtime.service import MAX_REQUEST_BYTES, default_socket_path


class RuntimeClient:
    def __init__(self, socket_path=None, timeout=10.0):
        self.socket_path = str(socket_path or default_socket_path())
        self.timeout = timeout

    def call(self, capability, args=None, timeout_ms=None, request_id=None):
        request = {
            "id": request_id or uuid.uuid4().hex,
            "capability": capability,
            "args": args or {},
        }
        if timeout_ms is not None:
            request["timeout_ms"] = timeout_ms
        payload = (json.dumps(request, ensure_ascii=False) + "\n").encode()
        if len(payload) > MAX_REQUEST_BYTES:
            raise ValueError("请求超过1MiB限制")

        # The transport must outlive the execution budget sent to Runtime.
        # Otherwise a valid long-running capability (for example adaptive
        # microphone capture) can finish just after the client has already
        # closed its socket. Keep a small allowance for dispatch and JSON I/O.
        response_timeout = self.timeout
        if timeout_ms is not None:
            response_timeout = max(response_timeout, timeout_ms / 1000 + 2.0)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(response_timeout)
            sock.connect(self.socket_path)
            sock.sendall(payload)
            with sock.makefile("rb") as reader:
                line = reader.readline(MAX_REQUEST_BYTES + 1)
        if not line:
            raise RuntimeError("Runtime未返回响应")
        if len(line) > MAX_REQUEST_BYTES:
            raise RuntimeError("Runtime响应超过1MiB限制")
        return json.loads(line)
