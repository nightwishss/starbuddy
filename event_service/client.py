"""Dependency-free client for the owner-only semantic event bus."""

from __future__ import annotations

import json
import os
import socket


DEFAULT_SOCKET = os.environ.get("VQ_EVENT_SOCKET", f"/run/user/{os.getuid()}/vq-events.sock")


class EventClientError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class EventClient:
    def __init__(self, socket_path: str = DEFAULT_SOCKET, *, timeout=25.0):
        self.socket_path = socket_path
        self.timeout = timeout

    def call(self, operation: str, **arguments) -> dict:
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
                if len(response) > 1024 * 1024:
                    raise EventClientError("response_too_large", "event response too large")
        try:
            value = json.loads(response)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise EventClientError("invalid_response", "invalid event response") from exc
        if not value.get("ok"):
            raise EventClientError(value.get("code", "service_error"), value.get("error", "event error"))
        return value["result"]

    def publish(self, topic: str, source: str, payload: dict, *, privacy="local", trace_id=None):
        values = {"topic": topic, "source": source, "payload": payload, "privacy": privacy}
        if trace_id is not None:
            values["trace_id"] = trace_id
        return self.call("publish", **values)

    def read(self, after_revision=0, *, topics=None, limit=64):
        return self.call("read", after_revision=after_revision, topics=topics, limit=limit)

    def wait(self, after_revision, *, topics=None, timeout=20.0, limit=64):
        return self.call("wait", after_revision=after_revision, topics=topics,
                         timeout=timeout, limit=limit)

    def stats(self):
        return self.call("stats")
