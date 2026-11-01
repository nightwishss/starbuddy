"""Loopback-only client for the hidden pose runner."""

import json
import urllib.request


class PoseClient:
    def __init__(self, base_url="http://127.0.0.1:8791", timeout=1.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def status(self):
        with urllib.request.urlopen(self.base_url + "/status", timeout=self.timeout) as response:
            return json.loads(response.read())

