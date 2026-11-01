"""JSONL Unix-socket client; media bytes move through private staging files."""

from __future__ import annotations

import json
import os
import shutil
import socket
from pathlib import Path


class MediaClientError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def default_socket_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return Path(runtime) / "vq-media.sock"


class MediaClient:
    def __init__(self, socket_path=None, timeout=10.0):
        self.socket_path = Path(socket_path or default_socket_path())
        self.timeout = timeout

    def call(self, op: str, **arguments):
        request = {"op": op, **arguments}
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout)
            client.connect(str(self.socket_path))
            client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
            line = client.makefile("rb").readline(65537)
        if not line or len(line) > 65536:
            raise MediaClientError("PROTOCOL_ERROR", "Media Service响应无效")
        response = json.loads(line)
        if not response.get("ok"):
            error = response.get("error") or {}
            raise MediaClientError(str(error.get("code", "MEDIA_ERROR")),
                                   str(error.get("message", "Media Service调用失败")))
        return response.get("result")

    def create_from_bytes(self, owner: str, mime_type: str, data: bytes, *, kind=None, ttl_seconds=None):
        begin = self.call("begin", owner=owner, mime_type=mime_type, kind=kind, ttl_seconds=ttl_seconds)
        upload_path = Path(begin["upload_path"])
        try:
            if len(data) > int(begin["max_bytes"]):
                raise MediaClientError("SIZE_LIMIT", "媒体大小超出限制")
            upload_path.write_bytes(data)
            upload_path.chmod(0o600)
            return self.call("commit", owner=owner, upload_id=begin["upload_id"])
        except Exception:
            try:
                self.call("abort", owner=owner, upload_id=begin["upload_id"])
            except Exception:
                pass
            raise

    def create_from_path(self, owner: str, mime_type: str, source: Path, *, kind=None, ttl_seconds=None):
        begin = self.call("begin", owner=owner, mime_type=mime_type, kind=kind, ttl_seconds=ttl_seconds)
        upload_path = Path(begin["upload_path"])
        try:
            source = Path(source)
            if source.stat().st_size > int(begin["max_bytes"]):
                raise MediaClientError("SIZE_LIMIT", "媒体大小超出限制")
            with source.open("rb") as reader, upload_path.open("wb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
            upload_path.chmod(0o600)
            return self.call("commit", owner=owner, upload_id=begin["upload_id"])
        except Exception:
            try:
                self.call("abort", owner=owner, upload_id=begin["upload_id"])
            except Exception:
                pass
            raise
