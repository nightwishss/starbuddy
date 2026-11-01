"""Small loopback client for the HDMI display state service."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class DisplayError(RuntimeError):
    pass


class DisplayClient:
    def __init__(self, endpoint: str = "http://127.0.0.1:8765/internal/state", timeout=0.4):
        if not endpoint.startswith("http://127.0.0.1:"):
            raise ValueError("display endpoint must be loopback")
        self.endpoint = endpoint
        self.timeout = timeout

    def update(self, **changes) -> dict:
        return self._post(self.endpoint, changes)["state"]

    def quiet_task(self, task_id):
        return self._post(self.endpoint.rsplit("/internal/", 1)[0] + "/internal/quiet-task", {"task_id": task_id}, timeout=2)

    def hug_idle(self, trace_id):
        return self._post(self.endpoint.rsplit("/internal/", 1)[0] + "/internal/hug-idle", {"trace_id": trace_id})

    def _post(self, endpoint, changes, *, timeout=None):
        body = json.dumps(changes, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            endpoint, data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout if timeout is None else timeout) as response:
                payload = json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise DisplayError("product display unavailable") from exc
        if not isinstance(payload, dict) or not payload.get("ok"):
            raise DisplayError("product display rejected update")
        return payload
