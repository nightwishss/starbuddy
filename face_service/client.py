"""Loopback client for semantic face presence state."""

from __future__ import annotations

import json
import urllib.request


class FaceClient:
    def __init__(self, base_url="http://127.0.0.1:8794", *, timeout=2.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def status(self):
        with urllib.request.urlopen(self.base_url + "/status", timeout=self.timeout) as response:
            return json.load(response)
