"""Owner-local JSONL client for semantic perception context."""

from __future__ import annotations

import json
import os
import socket


DEFAULT_SOCKET = os.environ.get(
    "VQ_PERCEPTION_CONTEXT_SOCKET", f"/run/user/{os.getuid()}/vq-perception-context.sock",
)


class PerceptionContextError(RuntimeError):
    pass


class PerceptionContextClient:
    def __init__(self, socket_path=DEFAULT_SOCKET, *, timeout=2.0):
        self.socket_path = socket_path
        self.timeout = timeout

    def call(self, operation, **arguments):
        request = json.dumps({"operation": operation, **arguments}, ensure_ascii=False) + "\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout)
            client.connect(self.socket_path)
            client.sendall(request.encode())
            response = bytearray()
            while not response.endswith(b"\n"):
                chunk = client.recv(65536)
                if not chunk:
                    break
                response.extend(chunk)
                if len(response) > 256 * 1024:
                    raise PerceptionContextError("response too large")
        try:
            value = json.loads(response)
        except (ValueError, UnicodeDecodeError) as exc:
            raise PerceptionContextError("invalid context response") from exc
        if not value.get("ok"):
            raise PerceptionContextError(value.get("error", "perception context error"))
        return value["result"]

    def snapshot(self):
        return self.call("context.snapshot")

    def observe(self, **observation):
        return self.call("context.observe", observation=observation)

    def stats(self):
        return self.call("context.stats")

    def fusion_snapshot(self, session_id=None):
        return self.call("fusion.snapshot", session_id=session_id)

    def snooze(self, session_id, seconds, *, echo=False):
        return self.call("fusion.snooze", session_id=session_id, seconds=seconds,
                         origin="user", echo=echo)

    def posture_proposal(self):
        return self.call("fusion.posture.proposal")

    def interaction_state(self):
        return self.call("fusion.interaction")

    def close(self):
        pass
