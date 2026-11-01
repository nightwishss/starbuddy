"""Client for model inventory and resource leases."""

from __future__ import annotations

import json
import os
import socket


DEFAULT_SOCKET = os.environ.get("VQ_CONTROL_SOCKET", f"/run/user/{os.getuid()}/vq-control.sock")


class ControlPlaneError(RuntimeError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code, self.details = code, details or {}


class ControlPlaneClient:
    def __init__(self, socket_path=DEFAULT_SOCKET, *, timeout=5):
        self.socket_path, self.timeout = socket_path, timeout

    def call(self, operation, **values):
        body = json.dumps({"operation": operation, **values}, ensure_ascii=False) + "\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout)
            client.connect(self.socket_path)
            client.sendall(body.encode())
            raw = client.makefile("rb").readline(1024 * 1024)
        value = json.loads(raw)
        if not value.get("ok"):
            raise ControlPlaneError(value.get("code", "service_error"),
                                    value.get("error", "control error"), value.get("details"))
        return value["result"]

    def models(self, task=None):
        return self.call("models.list", task=task)

    def model(self, model_id):
        return self.call("models.get", model_id=model_id)

    def acquire(self, owner, resources, purpose, *, priority=50, ttl_seconds=30):
        return self.call("resources.acquire", owner=owner, resources=resources, purpose=purpose,
                         priority=priority, ttl_seconds=ttl_seconds)

    def acquire_model(self, owner, model_id, *, priority=50, ttl_seconds=30):
        return self.call("models.acquire", owner=owner, model_id=model_id,
                         priority=priority, ttl_seconds=ttl_seconds)

    def renew(self, lease_id, owner, ttl_seconds=30):
        return self.call("resources.renew", lease_id=lease_id, owner=owner, ttl_seconds=ttl_seconds)

    def release(self, lease_id, owner):
        return self.call("resources.release", lease_id=lease_id, owner=owner)

    def status(self):
        return self.call("resources.status")

    def llm_enter(self, owner, *, kind="foreground", ttl_seconds=30):
        return self.call("llm.enter", owner=owner, kind=kind, ttl_seconds=ttl_seconds)

    def llm_finish(self, token, owner, *, terminal=True):
        return self.call("llm.finish", token=token, owner=owner, terminal=terminal)

    def llm_status(self):
        return self.call("llm.status")
