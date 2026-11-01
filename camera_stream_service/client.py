"""Loopback client exposing the CameraPreview-compatible application surface."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

from PIL import Image


class CameraStreamError(RuntimeError):
    pass


class CameraStreamClient:
    def __init__(self, base_url="http://127.0.0.1:8781", *, timeout=4.0,
                 consumer="display", rotation=180, source=None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.consumer = consumer
        self.vision_rotation = rotation
        self.source = source

    def _url(self, path, query=""):
        parts = []
        if query:
            parts.append(query)
        if self.source:
            from urllib.parse import quote
            parts.append("source=" + quote(self.source, safe=""))
        return self.base_url + path + (("?" + "&".join(parts)) if parts else "")

    def start(self):
        pass

    def pause(self):
        # Consumer-local pause only. The stream service stops hardware when its
        # actual HTTP subscribers and short frame leases reach zero.
        pass

    def close(self):
        pass

    def status(self):
        try:
            with urllib.request.urlopen(self._url("/status"), timeout=self.timeout) as response:
                value = json.load(response)
        except (OSError, urllib.error.URLError, ValueError) as exc:
            return {"enabled": False, "ready": False, "error": type(exc).__name__}
        value["enabled"] = bool(value.get("ready") or value.get("subscriber_count"))
        return value

    def wait_frame(self, after_sequence=0, timeout=3.0):
        request = urllib.request.Request(
            self._url("/frame.jpg", f"consumer={self.consumer}&after={int(after_sequence)}"),
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=max(self.timeout, timeout)) as response:
                frame = response.read(2 * 1024 * 1024 + 1)
                sequence = int(response.headers.get("X-VQ-Sequence", "0"))
        except (OSError, urllib.error.URLError, ValueError):
            return None
        if len(frame) > 2 * 1024 * 1024 or sequence <= after_sequence:
            return None
        return sequence, frame

    def frame_for_vision(self, frame: bytes) -> bytes:
        if self.vision_rotation == 0:
            return frame
        transpose = {
            90: Image.Transpose.ROTATE_270,
            180: Image.Transpose.ROTATE_180,
            270: Image.Transpose.ROTATE_90,
        }.get(self.vision_rotation)
        if transpose is None:
            raise CameraStreamError("unsupported camera rotation")
        with Image.open(io.BytesIO(frame)) as image:
            output = io.BytesIO()
            image.transpose(transpose).save(output, format="JPEG", quality=90)
            return output.getvalue()
