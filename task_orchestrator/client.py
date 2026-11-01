"""Dependency-free client for the owner-only Task Orchestrator service."""

from __future__ import annotations

import json
import hashlib
import os
import socket


DEFAULT_SOCKET = os.environ.get("VQ_TASK_ORCHESTRATOR_SOCKET",
                                f"/run/user/{os.getuid()}/vq-task-orchestrator.sock")


def request_task_id(session_id, request_id):
    """Stable lookup for a previously submitted idempotent create request."""
    key = json.dumps([session_id, request_id], separators=(",", ":"))
    return "task_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


class TaskClientError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class TaskClient:
    def __init__(self, socket_path=DEFAULT_SOCKET, *, timeout=45.0):
        self.socket_path = str(socket_path)
        self.timeout = timeout

    def call(self, operation: str, **arguments):
        request = json.dumps({"operation": operation, **arguments}, ensure_ascii=False) + "\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout)
            client.connect(self.socket_path)
            client.sendall(request.encode("utf-8"))
            response = bytearray()
            while not response.endswith(b"\n"):
                chunk = client.recv(65536)
                if not chunk:
                    break
                response.extend(chunk)
                if len(response) > 1024 * 1024:
                    raise TaskClientError("response_too_large", "task response too large")
        try:
            value = json.loads(response)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TaskClientError("invalid_response", "invalid task response") from exc
        if not value.get("ok"):
            raise TaskClientError(value.get("code", "service_error"), value.get("error", "task error"))
        return value["result"]

    def create(self, utterance: str, *, timezone=None, source_conversation_id=None,
               proposal=None, request_id=None):
        values = {"utterance": utterance}
        if timezone is not None:
            values["timezone"] = timezone
        if source_conversation_id is not None:
            values["source_conversation_id"] = source_conversation_id
        if proposal is not None:
            values["proposal"] = proposal
        if request_id is not None:
            values["request_id"] = request_id
        return self.call("task.create", **values)

    def get(self, task_id: str):
        return self.call("task.get", task_id=task_id)

    def list(self, *, states=None, limit=64):
        return self.call("task.list", states=states, limit=limit)

    def cancel(self, task_id: str):
        return self.call("task.cancel", task_id=task_id)

    def stats(self):
        return self.call("task.stats")
