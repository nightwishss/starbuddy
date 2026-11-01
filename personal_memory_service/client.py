"""Synchronous Unix-socket client for Personal Memory."""

import json
import os
import socket

DEFAULT_SOCKET = os.environ.get("VQ_PERSONAL_MEMORY_SOCKET", "/run/user/1000/vq-personal-memory.sock")
MAX_RESPONSE = 256 * 1024


class PersonalMemoryClientError(RuntimeError):
    pass


class PersonalMemoryClient:
    def __init__(self, socket_path=DEFAULT_SOCKET, timeout=3.0):
        self.socket_path, self.timeout = socket_path, timeout

    def call(self, operation, **values):
        payload = (json.dumps({"operation": operation, **values}, ensure_ascii=False) + "\n").encode()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout); client.connect(self.socket_path); client.sendall(payload)
            response = client.makefile("rb").readline(MAX_RESPONSE + 1)
        if not response or len(response) > MAX_RESPONSE:
            raise PersonalMemoryClientError("invalid personal-memory response")
        value = json.loads(response)
        if not value.get("ok"):
            raise PersonalMemoryClientError(f"{value.get('code')}: {value.get('error')}")
        return value["result"]

    def profile_get(self, profile_id): return self.call("profile.get", profile_id=profile_id)
    def profile_set(self, profile_id, field, value, *, consent):
        return self.call("profile.set", profile_id=profile_id, field=field, value=value, consent=consent)
    def recall(self, profile_id, query="", limit=6):
        return self.call("memory.recall", profile_id=profile_id, query=query, limit=limit)

