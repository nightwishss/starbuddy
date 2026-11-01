"""Phone Gateway adapter for the shared VQ Media Service."""

from __future__ import annotations

import os
from pathlib import Path

from media_service import MediaClient, MediaClientError

from .store import MediaError, MediaObject, MediaUnavailableError


class SharedMediaStoreAdapter:
    """Expose the legacy MediaStore interface over the shared media data plane."""

    LIMITS = {"audio": 16 * 1024 * 1024, "image": 16 * 1024 * 1024,
              "generated": 32 * 1024 * 1024}

    def __init__(self, client: MediaClient | None = None, work_root: Path | None = None):
        self.client = client or MediaClient()
        runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        self.root = Path(work_root or Path(runtime) / "vq-phone-gateway/work")
        self.root.mkdir(parents=True, exist_ok=True)
        self.root.chmod(0o700)

    def _object(self, value: dict, owner: str, path: Path | None = None) -> MediaObject:
        try:
            return MediaObject(
                media_id=str(value["media_id"]), owner=owner, kind=str(value["kind"]),
                mime_type=str(value["mime_type"]), size=int(value["size"]),
                created_at=float(value["created_at"]), expires_at=float(value["expires_at"]),
                path=path or Path("/dev/null"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MediaError("Media Service返回格式无效。") from exc

    def create(self, owner: str, mime_type: str, data: bytes, *, kind: str | None = None) -> MediaObject:
        try:
            value = self.client.create_from_bytes(owner, mime_type, data, kind=kind)
            return self._object(value, owner)
        except MediaClientError as exc:
            raise MediaError(str(exc)) from exc
        except OSError as exc:
            raise MediaUnavailableError("Media Service暂时不可用。") from exc

    def get(self, media_id: str, owner: str) -> MediaObject:
        try:
            value = self.client.call("info", owner=owner, media_id=media_id)
            resolved = self.client.call("resolve", owner=owner, media_id=media_id)
            return self._object(value, owner, Path(resolved["path"]))
        except MediaClientError as exc:
            raise MediaError(str(exc)) from exc
        except (OSError, KeyError) as exc:
            raise MediaUnavailableError("Media Service暂时不可用。") from exc

    def consume(self, media_id: str, owner: str) -> bytes:
        return self.get(media_id, owner).path.read_bytes()

    def release(self, media_id: str, owner: str) -> None:
        try:
            self.client.call("release", owner=owner, media_id=media_id)
        except MediaClientError as exc:
            raise MediaError(str(exc)) from exc
        except OSError as exc:
            raise MediaUnavailableError("Media Service暂时不可用。") from exc

    def cleanup(self, now: float | None = None) -> int:
        del now
        try:
            return int(self.client.call("cleanup")["removed"])
        except MediaClientError as exc:
            raise MediaError(str(exc)) from exc
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise MediaUnavailableError("Media Service暂时不可用。") from exc
