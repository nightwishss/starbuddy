"""Client for the consent-gated improvement audit service."""

from __future__ import annotations

import json
import os
import socket

DEFAULT_SOCKET = os.environ.get("VQ_IMPROVEMENT_SOCKET", f"/run/user/{os.getuid()}/vq-improvement.sock")


class ImprovementClientError(RuntimeError):
    def __init__(self, code, message): super().__init__(message); self.code = code


class ImprovementClient:
    def __init__(self, socket_path=DEFAULT_SOCKET, timeout=5): self.socket_path, self.timeout = socket_path, timeout
    def call(self, operation, **values):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout); client.connect(self.socket_path)
            client.sendall((json.dumps({"operation": operation, **values}, ensure_ascii=False) + "\n").encode())
            raw = client.makefile("rb").readline(1024 * 1024)
        result = json.loads(raw)
        if not result.get("ok"): raise ImprovementClientError(result.get("code", "error"), result.get("error", "error"))
        return result["result"]
    def stats(self): return self.call("stats")
