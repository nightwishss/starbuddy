"""Validated public feedback clips, not a recording library or TTS warm-up.

Only seven fixed, public product utterances are eligible. This class never
records, synthesizes, plays, resolves another media handle, or stores user text.
The imported generated clip belongs to the kiosk until explicitly handed off;
after handoff even a lost Output acknowledgement must not trigger its release.
The 120-second Media TTL bounds that uncertain-delivery case.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading

from interaction_context.situation import TASKS


KIOSK_OWNER = "session:display-kiosk"
BACKEND = "melo-qnn-htp"
MAX_MANIFEST_BYTES = 16 * 1024
MAX_WAV_BYTES = 2 * 1024 * 1024
MAX_OWNED_HANDLES = 8
MEDIA_TTL_SECONDS = 120
PUBLIC_TEXTS = frozenset(
    [row["hint"] for row in TASKS] + [
        "答对啦！要换一题，还是说说你是怎么想到的？",
        "这次还不对。你可以再试一下，也可以问我怎么想。",
        "我可以给你一个小提示，要听听吗？",
        "好，我先不提示。需要时你再叫我。",
    ]
)
_TEXT_KEYS = {hashlib.sha256(text.encode("utf-8")).hexdigest() for text in PUBLIC_TEXTS}
_FILE = re.compile(r"feedback-[0-9a-f]{16}\.wav\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_HANDLE = re.compile(r"generated_[A-Za-z0-9_-]{20,64}\Z")


def _no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest field")
        result[key] = value
    return result


def _signature(metadata):
    return (metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


def _open_root(path):
    """Traverse directory components without reading them or following links.

    Linux O_PATH needs only search permission on ancestors. O_RDONLY would
    additionally request directory read access and can fail under a service's
    filesystem sandbox even when the approved manifest/WAV leaves are readable.
    Leaf files are still opened O_RDONLY and fully validated below.
    """
    path = Path(os.path.abspath(path))
    flags = getattr(os, "O_PATH", os.O_RDONLY) | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_regular(root_fd, name, maximum):
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
    try:
        metadata = os.fstat(descriptor)
        if (not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1
                or not 0 < metadata.st_size <= maximum):
            raise ValueError("invalid public asset")
        return descriptor, metadata
    except BaseException:
        os.close(descriptor)
        raise


def _read_bound(descriptor, maximum):
    chunks = []
    remaining = maximum + 1
    while remaining:
        chunk = os.read(descriptor, min(remaining, 65536))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    value = b"".join(chunks)
    if len(value) > maximum:
        raise ValueError("public asset too large")
    return value


def _manifest(root_fd):
    descriptor, metadata = _open_regular(root_fd, "manifest.json", MAX_MANIFEST_BYTES)
    try:
        raw = _read_bound(descriptor, MAX_MANIFEST_BYTES)
        if len(raw) != metadata.st_size or _signature(metadata) != _signature(os.fstat(descriptor)):
            raise ValueError("manifest changed")
    finally:
        os.close(descriptor)
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates)
    if (type(value) is not dict
            or set(value) != {"schema_version", "speed", "backend", "entries"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or type(value["speed"]) not in (int, float) or value["speed"] != 1
            or value["backend"] != BACKEND
            or type(value["entries"]) is not dict or len(value["entries"]) > len(PUBLIC_TEXTS)):
        raise ValueError("invalid public manifest")
    for key, entry in value["entries"].items():
        if (type(key) is not str or key not in _TEXT_KEYS or type(entry) is not dict
                or set(entry) != {"file", "sha256", "size_bytes"}
                or type(entry["file"]) is not str or not _FILE.fullmatch(entry["file"])
                or type(entry["sha256"]) is not str or not _HASH.fullmatch(entry["sha256"])
                or type(entry["size_bytes"]) is not int
                or not 44 <= entry["size_bytes"] <= MAX_WAV_BYTES):
            raise ValueError("invalid public entry")
    return value["entries"]


def _validated_wav(root_fd, entry):
    descriptor, metadata = _open_regular(root_fd, entry["file"], MAX_WAV_BYTES)
    try:
        raw = _read_bound(descriptor, MAX_WAV_BYTES)
        if (len(raw) != entry["size_bytes"] or len(raw) != metadata.st_size
                or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE"
                or int.from_bytes(raw[4:8], "little") != len(raw) - 8
                or hashlib.sha256(raw).hexdigest() != entry["sha256"]
                or _signature(metadata) != _signature(os.fstat(descriptor))):
            raise ValueError("public wav validation failed")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor, metadata
    except BaseException:
        os.close(descriptor)
        raise


class FixedResponseAudio:
    """One bounded importer for a fixed public asset directory.

    ``asset_root`` is a host/test dependency, never a request field. Production
    defaults to this module's static/situated-feedback directory. Status returns
    the last verified public-asset count and never scans paths or Media objects.
    """

    def __init__(self, media, output, asset_root=None):
        self.media = media
        self.output = output
        self.asset_root = Path(asset_root) if asset_root is not None else Path(__file__).parent / "static" / "situated-feedback"
        self._owned = set()
        self._ready = set()
        self._verification_state = 'dependencies_unavailable'
        self._lock = threading.RLock()
        if self._available():
            self._verify_public_assets()

    def _available(self):
        return (callable(getattr(self.media, "create_from_path", None))
                and callable(getattr(self.media, "call", None))
                and callable(getattr(self.output, "play_media", None)))

    def _verify_public_assets(self):
        root_fd = None
        stage = 'root'
        try:
            root_fd = _open_root(self.asset_root)
            stage = 'manifest'
            entries = _manifest(root_fd)
            stage = 'wave'
            for key, entry in entries.items():
                descriptor = None
                try:
                    descriptor, _metadata = _validated_wav(root_fd, entry)
                    self._ready.add(key)
                except (OSError, ValueError, TypeError) as error:
                    self._verification_state = stage + ':' + type(error).__name__ + ':' + str(getattr(error, 'errno', 0))
                finally:
                    if descriptor is not None:
                        os.close(descriptor)
            if len(self._ready) == len(entries):
                self._verification_state = 'verified'
        except (OSError, ValueError, TypeError) as error:
            self._verification_state = stage + ':' + type(error).__name__ + ':' + str(getattr(error, 'errno', 0))
            self._ready.clear()
        finally:
            if root_fd is not None:
                os.close(root_fd)

    def status(self):
        with self._lock:
            return {"ready_public_assets": len(self._ready),
                    "verification_state": self._verification_state,
                    "dependencies_available": self._available()}

    def prepare(self, text, speed=1.0):
        if (type(text) is not str or text not in PUBLIC_TEXTS
                or type(speed) not in (int, float) or speed != 1.0 or not self._available()):
            return None
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        with self._lock:
            if len(self._owned) >= MAX_OWNED_HANDLES:
                return None
            root_fd = descriptor = None
            imported = None
            try:
                root_fd = _open_root(self.asset_root)
                entries = _manifest(root_fd)
                self._ready.intersection_update(entries)
                entry = entries.get(key)
                if entry is None:
                    self._ready.discard(key)
                    return None
                descriptor, metadata = _validated_wav(root_fd, entry)
                self._ready.add(key)
                # This fd was opened only under the fixed public asset root.
                # Passing its pinned view prevents a pathname/symlink swap
                # between hash validation and MediaClient's bounded copy.
                fd_root = Path("/proc/self/fd") if Path("/proc/self/fd").is_dir() else Path("/dev/fd")
                reply = self.media.create_from_path(
                    KIOSK_OWNER, "audio/wav", fd_root / str(descriptor),
                    kind="generated", ttl_seconds=MEDIA_TTL_SECONDS,
                )
                if type(reply) is not dict or type(reply.get("media_id")) is not str or not _HANDLE.fullmatch(reply["media_id"]):
                    return None
                imported = reply["media_id"]
                if imported in self._owned:
                    # A fresh create must not alias an earlier in-flight clip.
                    imported = None
                    return None
                self._owned.add(imported)
                if (reply.get("kind") != "generated" or reply.get("mime_type") != "audio/wav"
                        or type(reply.get("size")) is not int or reply["size"] != entry["size_bytes"]
                        or _signature(metadata) != _signature(os.fstat(descriptor))):
                    self.release(imported)
                    return None
                return imported
            except Exception:
                self._ready.discard(key)
                if imported is not None:
                    self.release(imported)
                return None
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                if root_fd is not None:
                    os.close(root_fd)

    def release(self, handle):
        """Release only our own, not-yet-handed-off generated clip."""
        with self._lock:
            if type(handle) is not str or handle not in self._owned:
                return False
            try:
                reply = self.media.call("release", owner=KIOSK_OWNER, media_id=handle)
            except Exception:
                # Uncertain cleanup retains ownership and its slot. Repeated
                # failures cannot create an unbounded set of pending imports.
                return False
            # MediaServer.release returns this exact positive receipt. A
            # missing, false or mismatched acknowledgement is not cleanup.
            if (type(reply) is not dict or reply.get("released") is not True
                    or reply.get("media_id") != handle):
                return False
            self._owned.remove(handle)
            return True

    def handoff(self, handle):
        """Transfer once before Output dispatch; never release after transfer."""
        with self._lock:
            if type(handle) is not str or handle not in self._owned:
                return False
            self._owned.remove(handle)
            return True
