"""Loopback client used by UI and non-UI applications."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class PostureClientError(RuntimeError): pass


class PostureClient:
    def __init__(self, base_url="http://127.0.0.1:8792/api/posture", timeout=4):
        self.base_url, self.timeout = base_url.rstrip("/"), timeout
    def _call(self, operation, method="POST", arguments=None):
        request = urllib.request.Request(
            f"{self.base_url}/{operation}", method=method,
            data=None if method == "GET" else json.dumps(arguments or {},separators=(",", ":")).encode(),
            headers={} if method == "GET" else {"Content-Type":"application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response: value=json.load(response)
        except (OSError, urllib.error.URLError, ValueError) as exc:
            raise PostureClientError("posture service unavailable") from exc
        if not isinstance(value, dict): raise PostureClientError("invalid posture response")
        return value
    def status(self): return self._call("status", "GET")
    def start(self): return self._call("start")
    def pause(self): return self._call("pause")
    def calibrate(self): return self._call("calibrate")
    def reset(self): return self._call("reset")
    def simulate(self): return self._call("simulate")
    def monitor(self,task_id,*,duration_seconds,bad_duration_seconds=20,cooldown_seconds=180):
        return self._call("monitor",arguments={"task_id":task_id,"duration_seconds":duration_seconds,
            "bad_duration_seconds":bad_duration_seconds,"cooldown_seconds":cooldown_seconds})
    def stop(self, task_id): return self._call("stop", arguments={"task_id": task_id})
    def close(self): pass
