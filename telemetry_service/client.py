"""Unix-socket client for bounded telemetry snapshots."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path


class TelemetryClientError(RuntimeError):
    pass


def default_socket_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return Path(runtime) / "vq-telemetry.sock"


class TelemetryClient:
    def __init__(self, socket_path=None, timeout=2.0):
        self.socket_path = Path(socket_path or default_socket_path())
        self.timeout = timeout

    def call(self, op: str, **arguments):
        request = {"op": op, **arguments}
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(self.timeout)
                client.connect(str(self.socket_path))
                client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
                line = client.makefile("rb").readline(4 * 1024 * 1024 + 1)
        except OSError as exc:
            raise TelemetryClientError("telemetry service unavailable") from exc
        if not line or len(line) > 4 * 1024 * 1024:
            raise TelemetryClientError("invalid telemetry response")
        response = json.loads(line)
        if not response.get("ok"):
            raise TelemetryClientError(str((response.get("error") or {}).get("message", "failed")))
        return response.get("result")

    def latest(self):
        return self.call("latest")

    def history(self, limit=120):
        return self.call("history", limit=limit)
