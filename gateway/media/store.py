"""Small ephemeral media registry shared by phone, speech, and vision services."""

from __future__ import annotations

import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


class MediaError(ValueError):
    pass


class MediaUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class MediaObject:
    media_id: str
    owner: str
    kind: str
    mime_type: str
    size: int
    created_at: float
    expires_at: float
    path: Path

    def public(self) -> dict:
        value = asdict(self)
        value.pop("owner")
        value.pop("path")
        return value


class MediaStore:
    """Process-local metadata with mode-0600 files in an ephemeral directory."""

    LIMITS = {"audio": 10 * 1024 * 1024, "image": 8 * 1024 * 1024, "generated": 16 * 1024 * 1024}
    MIME = {
        "audio/webm": ("audio", b"\x1a\x45\xdf\xa3", None, ".webm"),
        "audio/ogg": ("audio", b"OggS", None, ".ogg"),
        "audio/wav": ("audio", b"RIFF", b"WAVE", ".wav"),
        "image/jpeg": ("image", b"\xff\xd8", b"\xff\xd9", ".jpg"),
    }

    def __init__(self, root: Path, ttl_seconds: int = 600, global_limit: int = 1024 * 1024 * 1024):
        self.root = root
        self.ttl_seconds = ttl_seconds
        self.global_limit = global_limit
        self._objects: dict[str, MediaObject] = {}
        self._lock = threading.RLock()
        root.mkdir(parents=True, exist_ok=True)
        root.chmod(0o700)
        # Sessions are process-local, so no media object can remain authorized
        # after a gateway restart. Remove orphaned files from this dedicated root.
        for path in root.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)

    def create(self, owner: str, mime_type: str, data: bytes, *, kind: str | None = None) -> MediaObject:
        spec = self.MIME.get(mime_type)
        if spec is None:
            raise MediaError("不支持的媒体类型。")
        expected_kind, prefix, suffix, extension = spec
        kind = kind or expected_kind
        if kind not in {expected_kind, "generated"}:
            raise MediaError("媒体用途与内容类型不匹配。")
        limit = self.LIMITS[kind]
        if not data or len(data) > limit:
            raise MediaError("媒体大小超出限制。")
        if not data.startswith(prefix) or (mime_type == "audio/wav" and data[8:12] != suffix) or (
            mime_type == "image/jpeg" and not data.endswith(suffix)
        ):
            raise MediaError("媒体文件头或结尾无效。")
        now = time.time()
        with self._lock:
            self.cleanup(now)
            if sum(item.size for item in self._objects.values()) + len(data) > self.global_limit:
                raise MediaError("媒体缓存空间不足。")
            media_id = f"{kind}_{secrets.token_urlsafe(18)}"
            path = self.root / f"{media_id}{extension}"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                path.unlink(missing_ok=True)
                raise
            item = MediaObject(media_id, owner, kind, mime_type, len(data), now, now + self.ttl_seconds, path)
            self._objects[media_id] = item
            return item

    def get(self, media_id: str, owner: str) -> MediaObject:
        with self._lock:
            item = self._objects.get(media_id)
            if item is None or item.expires_at <= time.time():
                if item is not None:
                    self._delete(item)
                raise MediaError("媒体不存在或已经过期。")
            if not secrets.compare_digest(item.owner, owner):
                raise MediaError("无权访问该媒体。")
            return item

    def consume(self, media_id: str, owner: str) -> bytes:
        item = self.get(media_id, owner)
        return item.path.read_bytes()

    def release(self, media_id: str, owner: str) -> None:
        item = self.get(media_id, owner)
        with self._lock:
            self._delete(item)

    def cleanup(self, now: float | None = None) -> int:
        now = time.time() if now is None else now
        removed = 0
        with self._lock:
            for item in list(self._objects.values()):
                if item.expires_at <= now:
                    self._delete(item)
                    removed += 1
        return removed

    def _delete(self, item: MediaObject) -> None:
        self._objects.pop(item.media_id, None)
        item.path.unlink(missing_ok=True)
