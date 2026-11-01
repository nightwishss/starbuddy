"""Client for the local output arbiter."""

from __future__ import annotations

import json
import os
import socket


DEFAULT_SOCKET = os.environ.get("VQ_OUTPUT_SOCKET", f"/run/user/{os.getuid()}/vq-output.sock")


class OutputClientError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class OutputClient:
    def __init__(self, socket_path=DEFAULT_SOCKET, *, timeout=35.0):
        self.socket_path = socket_path
        self.timeout = timeout

    def call(self, operation, **arguments):
        body = json.dumps({"operation": operation, **arguments}, ensure_ascii=False) + "\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout)
            client.connect(self.socket_path)
            client.sendall(body.encode())
            response = bytearray()
            while not response.endswith(b"\n"):
                chunk = client.recv(65536)
                if not chunk:
                    break
                response.extend(chunk)
                if len(response) > 256 * 1024:
                    raise OutputClientError("response_too_large", "output response too large")
        try:
            value = json.loads(response)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OutputClientError("invalid_response", "invalid output response") from exc
        if not value.get("ok"):
            raise OutputClientError(value.get("code", "service_error"), value.get("error", "output error"))
        return value["result"]

    def say(self, text: str, owner: str, *, category="conversation", volume_percent=70,
            expires_in_seconds=None, trace_id=None, guard=None, speed=1.0):
        values = {"text": text, "owner": owner, "category": category,
                  "volume_percent": volume_percent}
        if expires_in_seconds is not None:
            values["expires_in_seconds"] = expires_in_seconds
        if trace_id is not None:
            values["trace_id"] = trace_id
        if guard is not None:
            values["guard"] = guard
        if speed != 1.0 or type(speed) not in (int, float):
            values["speed"] = speed
        return self.call("say", **values)

    def play_media(self, media_id: str, owner: str, *, category="conversation", volume_percent=70,
                   expires_in_seconds=None, trace_id=None):
        values = {"media_id": media_id, "owner": owner, "category": category,
                  "volume_percent": volume_percent}
        if expires_in_seconds is not None:
            values["expires_in_seconds"] = expires_in_seconds
        if trace_id is not None:
            values["trace_id"] = trace_id
        return self.call("play_media", **values)

    def status(self, job_id: str):
        return self.call("status", job_id=job_id)

    def cancel(self, job_id: str):
        return self.call("cancel", job_id=job_id)

    def system_status(self):
        return self.call("system.status")
