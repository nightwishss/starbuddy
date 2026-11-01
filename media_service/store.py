"""Restart-recoverable ephemeral media objects with bounded ownership."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path


class MediaError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


_OWNER = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_MEDIA_ID = re.compile(r"^(?:audio|image|generated)_[A-Za-z0-9_-]{20,64}$")
_UPLOAD_ID = re.compile(r"^upload_[A-Za-z0-9_-]{20,64}$")


@dataclass(frozen=True)
class MediaObject:
    media_id: str
    owner_hash: str
    kind: str
    mime_type: str
    size: int
    created_at: float
    expires_at: float
    filename: str

    def public(self) -> dict:
        value = asdict(self)
        value.pop("owner_hash")
        value.pop("filename")
        return value


@dataclass(frozen=True)
class Upload:
    upload_id: str
    owner_hash: str
    kind: str
    mime_type: str
    created_at: float
    ttl_seconds: int
    path: Path


class SharedMediaStore:
    LIMITS = {
        "audio": 16 * 1024 * 1024,
        "image": 16 * 1024 * 1024,
        "generated": 32 * 1024 * 1024,
    }
    MIME = {
        "audio/webm": ("audio", b"\x1a\x45\xdf\xa3", None, ".webm"),
        "audio/ogg": ("audio", b"OggS", None, ".ogg"),
        "audio/wav": ("audio", b"RIFF", b"WAVE", ".wav"),
        "image/jpeg": ("image", b"\xff\xd8", b"\xff\xd9", ".jpg"),
        "image/png": ("image", b"\x89PNG\r\n\x1a\n", None, ".png"),
    }

    def __init__(self, root: Path, *, default_ttl=600, max_ttl=3600,
                 global_limit=1024 * 1024 * 1024, max_objects=256):
        self.root = Path(root)
        self.objects_root = self.root / "objects"
        self.staging_root = self.root / "staging"
        self.default_ttl = int(default_ttl)
        self.max_ttl = int(max_ttl)
        self.global_limit = int(global_limit)
        self.max_objects = int(max_objects)
        self._objects: dict[str, MediaObject] = {}
        self._uploads: dict[str, Upload] = {}
        self._lock = threading.RLock()
        for path in (self.root, self.objects_root, self.staging_root):
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(0o700)
        self._recover()

    @staticmethod
    def owner_hash(owner: str) -> str:
        if not isinstance(owner, str) or not _OWNER.fullmatch(owner):
            raise MediaError("INVALID_OWNER", "媒体所有者格式无效")
        return hashlib.sha256(owner.encode()).hexdigest()

    def begin(self, owner: str, mime_type: str, *, kind=None, ttl_seconds=None) -> dict:
        owner_hash = self.owner_hash(owner)
        spec = self.MIME.get(mime_type)
        if spec is None:
            raise MediaError("UNSUPPORTED_MEDIA_TYPE", "不支持的媒体类型")
        expected_kind = spec[0]
        kind = kind or expected_kind
        if kind not in {expected_kind, "generated"}:
            raise MediaError("KIND_MISMATCH", "媒体用途与内容类型不匹配")
        ttl = self.default_ttl if ttl_seconds is None else int(ttl_seconds)
        if ttl < 1 or ttl > self.max_ttl:
            raise MediaError("INVALID_TTL", "媒体TTL超出限制")
        upload_id = f"upload_{secrets.token_urlsafe(18)}"
        path = self.staging_root / f".{upload_id}.part"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(fd)
        upload = Upload(upload_id, owner_hash, kind, mime_type, time.time(), ttl, path)
        with self._lock:
            self._uploads[upload_id] = upload
        return {"upload_id": upload_id, "upload_path": str(path),
                "max_bytes": self.LIMITS[kind], "expires_in": ttl}

    def commit(self, upload_id: str, owner: str) -> MediaObject:
        if not isinstance(upload_id, str) or not _UPLOAD_ID.fullmatch(upload_id):
            raise MediaError("INVALID_UPLOAD", "上传句柄格式无效")
        owner_hash = self.owner_hash(owner)
        with self._lock:
            upload = self._uploads.get(upload_id)
            if upload is None:
                raise MediaError("UPLOAD_NOT_FOUND", "上传不存在或已经失效")
            if not secrets.compare_digest(upload.owner_hash, owner_hash):
                raise MediaError("PERMISSION_DENIED", "无权提交该媒体")
            self.cleanup()
            self._validate_file(upload)
            size = upload.path.stat().st_size
            if len(self._objects) >= self.max_objects:
                raise MediaError("QUOTA_EXCEEDED", "媒体对象数量达到上限")
            if sum(item.size for item in self._objects.values()) + size > self.global_limit:
                raise MediaError("QUOTA_EXCEEDED", "媒体缓存空间不足")
            extension = self.MIME[upload.mime_type][3]
            media_id = f"{upload.kind}_{secrets.token_urlsafe(18)}"
            filename = f"{media_id}{extension}"
            final_path = self.objects_root / filename
            upload.path.replace(final_path)
            os.chmod(final_path, 0o600)
            now = time.time()
            item = MediaObject(media_id, owner_hash, upload.kind, upload.mime_type,
                               size, now, now + upload.ttl_seconds, filename)
            try:
                self._write_metadata(item)
            except Exception:
                final_path.unlink(missing_ok=True)
                raise
            self._uploads.pop(upload_id, None)
            self._objects[media_id] = item
            return item

    def abort(self, upload_id: str, owner: str) -> None:
        owner_hash = self.owner_hash(owner)
        with self._lock:
            upload = self._uploads.get(upload_id)
            if upload is None:
                return
            if not secrets.compare_digest(upload.owner_hash, owner_hash):
                raise MediaError("PERMISSION_DENIED", "无权取消该媒体")
            self._uploads.pop(upload_id, None)
            upload.path.unlink(missing_ok=True)

    def get(self, media_id: str, owner: str) -> MediaObject:
        if not isinstance(media_id, str) or not _MEDIA_ID.fullmatch(media_id):
            raise MediaError("INVALID_MEDIA_ID", "媒体ID格式无效")
        owner_hash = self.owner_hash(owner)
        with self._lock:
            item = self._objects.get(media_id)
            if item is None or item.expires_at <= time.time():
                if item is not None:
                    self._delete(item)
                raise MediaError("MEDIA_NOT_FOUND", "媒体不存在或已经过期")
            if not secrets.compare_digest(item.owner_hash, owner_hash):
                raise MediaError("PERMISSION_DENIED", "无权访问该媒体")
            return item

    def resolve(self, media_id: str, owner: str) -> Path:
        item = self.get(media_id, owner)
        path = self.objects_root / item.filename
        if path.parent.resolve() != self.objects_root.resolve() or not path.is_file():
            self._delete(item)
            raise MediaError("MEDIA_NOT_FOUND", "媒体文件不存在")
        return path

    def release(self, media_id: str, owner: str) -> None:
        item = self.get(media_id, owner)
        with self._lock:
            self._delete(item)

    def list_owned(self, owner: str) -> list[MediaObject]:
        owner_hash = self.owner_hash(owner)
        with self._lock:
            self.cleanup()
            return sorted((item for item in self._objects.values()
                           if secrets.compare_digest(item.owner_hash, owner_hash)),
                          key=lambda item: item.created_at)

    def cleanup(self, now=None) -> int:
        now = time.time() if now is None else now
        removed = 0
        with self._lock:
            for item in list(self._objects.values()):
                if item.expires_at <= now:
                    self._delete(item)
                    removed += 1
            for upload in list(self._uploads.values()):
                if upload.created_at + 60 <= now:
                    self._uploads.pop(upload.upload_id, None)
                    upload.path.unlink(missing_ok=True)
            for path in self.staging_root.iterdir():
                try:
                    if path.is_file() and path.stat().st_mtime + 60 <= now:
                        path.unlink(missing_ok=True)
                except OSError:
                    continue
        return removed

    def _validate_file(self, upload: Upload) -> None:
        try:
            size = upload.path.stat().st_size
            limit = self.LIMITS[upload.kind]
            if size <= 0 or size > limit:
                raise MediaError("SIZE_LIMIT", "媒体大小超出限制")
            with upload.path.open("rb") as stream:
                head = stream.read(12)
                tail = b""
                if size >= 2:
                    stream.seek(-2, os.SEEK_END)
                    tail = stream.read(2)
        except OSError as exc:
            raise MediaError("UPLOAD_NOT_FOUND", "上传文件不可读") from exc
        _, prefix, suffix, _ = self.MIME[upload.mime_type]
        valid = head.startswith(prefix)
        if upload.mime_type == "audio/wav":
            valid = valid and head[8:12] == suffix
        elif upload.mime_type == "image/jpeg":
            valid = valid and tail == suffix
        if not valid:
            raise MediaError("INVALID_SIGNATURE", "媒体文件签名无效")

    def _metadata_path(self, media_id: str) -> Path:
        return self.objects_root / f".{media_id}.json"

    def _write_metadata(self, item: MediaObject) -> None:
        path = self._metadata_path(item.media_id)
        temp = self.objects_root / f".{item.media_id}.{secrets.token_hex(6)}.tmp"
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(asdict(item), stream, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            temp.replace(path)
        except Exception:
            temp.unlink(missing_ok=True)
            raise

    def _delete(self, item: MediaObject) -> None:
        self._objects.pop(item.media_id, None)
        (self.objects_root / item.filename).unlink(missing_ok=True)
        self._metadata_path(item.media_id).unlink(missing_ok=True)

    def _recover(self) -> None:
        now = time.time()
        referenced = set()
        for metadata in self.objects_root.glob(".*.json"):
            try:
                value = json.loads(metadata.read_text(encoding="utf-8"))
                item = MediaObject(**value)
                path = self.objects_root / item.filename
                if (not _MEDIA_ID.fullmatch(item.media_id) or item.expires_at <= now
                        or path.parent.resolve() != self.objects_root.resolve()
                        or not path.is_file() or path.stat().st_size != item.size):
                    raise ValueError("invalid media metadata")
                self._objects[item.media_id] = item
                referenced.add(item.filename)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                metadata.unlink(missing_ok=True)
        for path in self.objects_root.iterdir():
            if path.is_file() and not path.name.startswith(".") and path.name not in referenced:
                path.unlink(missing_ok=True)
            elif path.is_file() and path.name.startswith(".") and not path.name.endswith(".json"):
                path.unlink(missing_ok=True)
        for path in self.staging_root.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
