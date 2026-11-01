"""UVC camera discovery and bounded one-shot frame capture.

Control data crosses Runtime as JSON. Pixel data stays in a mode-0600 ephemeral
frame store and is referenced by an opaque ``frame_id``. Continuous streaming
and durable recordings intentionally remain outside this first-stage contract.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import secrets
import struct
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from device_management import HardwareProfileManager
from media_service import MediaClient, MediaClientError
from runtime.errors import HardwareError, ResourceNotFoundError, SchemaValidationError

_BY_ID_ROOT = Path("/dev/v4l/by-id")
_RUNTIME_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
_FRAME_ROOT = _RUNTIME_DIR / "vq-camera" / "frames"
_FRAME_TTL_SECONDS = 600
_MAX_FRAMES = 32
_MAX_FRAME_BYTES = 16 * 1024 * 1024
_FRAME_ID = re.compile(r"^frame_[A-Za-z0-9_-]{20,64}$")
_SIZE = re.compile(r"(\d{2,5})x(\d{2,5})")
_FORMAT_LINE = re.compile(r"(?:Compressed|Raw)\s*:\s*([a-zA-Z0-9_]+)\s*:\s*([^:]+):\s*(.*)$")


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    media_id: str
    camera_id: str
    mime_type: str
    width: int
    height: int
    source: dict
    transform: dict
    size: int
    created_at: float
    expires_at: float

    def public(self):
        value = asdict(self)
        value.pop("created_at")
        return value


_FRAMES: dict[str, FrameRecord] = {}
_LOCK = threading.RLock()
_STORE_READY = False
_MEDIA_OWNER = "service:camera-runtime"


def _media_client():
    return MediaClient()


def _camera_id(stable_name: str) -> str:
    digest = hashlib.sha256(stable_name.encode()).hexdigest()[:12]
    return f"camera-{digest}"


def _friendly_name(stable_name: str) -> str:
    name = stable_name
    if name.startswith("usb-"):
        name = name[4:]
    if name.endswith("-video-index0"):
        name = name[:-13]
    return name.replace("_", " ")[:160]


def _discover() -> dict[str, dict]:
    cameras = {}
    if not _BY_ID_ROOT.is_dir():
        return cameras
    for link in sorted(_BY_ID_ROOT.glob("*-video-index0")):
        try:
            device = link.resolve(strict=True)
        except OSError:
            continue
        if not device.name.startswith("video"):
            continue
        stable_name = link.name
        camera_id = _camera_id(stable_name)
        cameras[camera_id] = {
            "camera_id": camera_id,
            "name": _friendly_name(stable_name),
            "device": str(device),
            "stable_path": str(link),
            "connected": True,
        }
    return cameras


def _require_camera(camera_id: str) -> dict:
    camera = _discover().get(camera_id)
    if camera is None:
        raise ResourceNotFoundError(
            f"摄像头不存在或已经断开: {camera_id}",
            capability="camera", details={"camera_id": camera_id})
    return camera


def _camera_from_args(args: dict) -> dict:
    camera_id = args.get("camera_id")
    role = args.get("role")
    if bool(camera_id) == bool(role):
        raise SchemaValidationError(
            "camera_id和role必须且只能提供一个", capability="camera.snapshot")
    if camera_id:
        return _require_camera(camera_id)
    if not isinstance(role, str) or not role.startswith("camera."):
        raise SchemaValidationError("摄像头逻辑角色无效", capability="camera.snapshot")
    resolved = HardwareProfileManager().resolve(role)
    device = resolved.get("device") or {}
    stable_id = device.get("stable_id")
    camera = _discover().get(_camera_id(stable_id)) if isinstance(stable_id, str) else None
    if camera is None:
        raise ResourceNotFoundError(
            f"摄像头逻辑角色已经断开: {role}", capability="camera.snapshot")
    return camera


def _run_format_probe(device: str) -> list[dict]:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "v4l2", "-list_formats", "all", "-i", device],
            capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HardwareError(
            "无法查询摄像头格式", capability="camera",
            details={"reason": type(exc).__name__}) from exc
    formats = []
    for line in (result.stderr + "\n" + result.stdout).splitlines():
        match = _FORMAT_LINE.search(line)
        if not match:
            continue
        pixel_format, description, size_text = match.groups()
        sizes = []
        seen = set()
        for width, height in _SIZE.findall(size_text):
            pair = (int(width), int(height))
            if pair not in seen:
                seen.add(pair)
                sizes.append({"width": pair[0], "height": pair[1]})
        formats.append({
            "pixel_format": pixel_format.lower(),
            "description": description.strip()[:100],
            "sizes": sizes,
        })
    if not formats:
        raise HardwareError("摄像头未返回可用格式", capability="camera")
    return formats


def _ioc(direction, ioctl_type, number, size):
    return (direction << 30) | (ord(ioctl_type) << 8) | number | (size << 16)


_VIDIOC_QUERYCTRL = _ioc(3, "V", 36, 68)
_VIDIOC_G_CTRL = _ioc(3, "V", 27, 8)
_NEXT_CTRL = 0x80000000
_CTRL_DISABLED = 0x0001


def _query_controls(device: str) -> list[dict]:
    controls = []
    try:
        fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
    except OSError as exc:
        raise HardwareError("无法打开摄像头控制接口", capability="camera") from exc
    try:
        control_id = _NEXT_CTRL
        seen = set()
        while len(controls) < 128:
            query = bytearray(struct.pack("<II32siiiiIII", control_id, 0, b"", 0, 0, 0, 0, 0, 0, 0))
            try:
                fcntl.ioctl(fd, _VIDIOC_QUERYCTRL, query, True)
            except OSError:
                break
            values = struct.unpack("<II32siiiiIII", query)
            query_id, ctrl_type, raw_name, minimum, maximum, step, default, flags = values[:8]
            if query_id in seen:
                break
            seen.add(query_id)
            control_id = query_id | _NEXT_CTRL
            if flags & _CTRL_DISABLED or ctrl_type == 6:
                continue
            current = None
            value = bytearray(struct.pack("<Ii", query_id, 0))
            try:
                fcntl.ioctl(fd, _VIDIOC_G_CTRL, value, True)
                current = struct.unpack("<Ii", value)[1]
            except OSError:
                pass
            controls.append({
                "id": f"0x{query_id:08x}",
                "name": raw_name.split(b"\0", 1)[0].decode(errors="replace")[:100],
                "type": ctrl_type,
                "minimum": minimum,
                "maximum": maximum,
                "step": step,
                "default": default,
                "current": current,
                "read_only": bool(flags & 0x0004),
                "inactive": bool(flags & 0x0010),
            })
    finally:
        os.close(fd)
    return controls


def list_cameras(args=None):
    _ensure_store()
    cameras = []
    for camera in _discover().values():
        item = dict(camera)
        try:
            formats = _run_format_probe(camera["device"])
            item["formats"] = [entry["pixel_format"] for entry in formats]
            item["max_width"] = max(size["width"] for entry in formats for size in entry["sizes"])
            item["max_height"] = max(size["height"] for entry in formats for size in entry["sizes"])
        except (HardwareError, ValueError):
            item.update({"formats": [], "max_width": 0, "max_height": 0})
        cameras.append(item)
    return {"cameras": cameras, "count": len(cameras)}


def get_info(args):
    _ensure_store()
    camera = _require_camera(args["camera_id"])
    return {
        **camera,
        "formats": _run_format_probe(camera["device"]),
        "controls": _query_controls(camera["device"]),
    }


def _select_source(formats: list[dict], requested: dict) -> dict:
    pixel_format = requested.get("pixel_format", "mjpeg")
    chosen = next((entry for entry in formats if entry["pixel_format"] == pixel_format), None)
    if chosen is None:
        raise SchemaValidationError(
            f"摄像头不支持输入像素格式: {pixel_format}", capability="camera.snapshot")
    sizes = {(item["width"], item["height"]) for item in chosen["sizes"]}
    if not sizes:
        raise HardwareError("摄像头格式没有可用分辨率", capability="camera.snapshot")
    if "width" in requested or "height" in requested:
        if "width" not in requested or "height" not in requested:
            raise SchemaValidationError(
                "source.width和source.height必须同时提供", capability="camera.snapshot")
        size = (requested["width"], requested["height"])
        if size not in sizes:
            raise SchemaValidationError(
                f"摄像头不支持{pixel_format} {size[0]}x{size[1]}",
                capability="camera.snapshot", details={"supported": sorted(sizes)})
    else:
        preferred = [(1920, 1080), (1280, 720), (640, 480)]
        size = next((item for item in preferred if item in sizes), max(sizes, key=lambda item: item[0] * item[1]))
    return {
        "width": size[0], "height": size[1],
        "pixel_format": pixel_format, "fps": requested.get("fps", 30),
    }


def _validate_transform(source: dict, transform: dict) -> tuple[list[str], int, int, dict]:
    width, height = source["width"], source["height"]
    filters = []
    normalized = {}
    crop = transform.get("crop")
    if crop:
        if crop["x"] + crop["width"] > width or crop["y"] + crop["height"] > height:
            raise SchemaValidationError(
                "crop区域超出源图像边界", capability="camera.snapshot",
                details={"source_width": width, "source_height": height})
        filters.append(f"crop={crop['width']}:{crop['height']}:{crop['x']}:{crop['y']}")
        width, height = crop["width"], crop["height"]
        normalized["crop"] = dict(crop)
    rotation = transform.get("rotation", 0)
    if rotation == 90:
        filters.append("transpose=clock")
        width, height = height, width
    elif rotation == 180:
        filters.extend(["transpose=clock", "transpose=clock"])
    elif rotation == 270:
        filters.append("transpose=cclock")
        width, height = height, width
    normalized["rotation"] = rotation
    resize = transform.get("resize")
    if resize:
        filters.append(f"scale={resize['width']}:{resize['height']}:flags=lanczos")
        width, height = resize["width"], resize["height"]
        normalized["resize"] = dict(resize)
    return filters, width, height, normalized


def _cleanup(now=None):
    now = time.time() if now is None else now
    for frame in list(_FRAMES.values()):
        if frame.expires_at <= now:
            _delete_frame(frame)


def _ensure_store():
    global _STORE_READY
    with _LOCK:
        if _STORE_READY:
            return
        _FRAME_ROOT.mkdir(parents=True, exist_ok=True)
        _FRAME_ROOT.chmod(0o700)
        # Registrations are process-local. A Runtime restart therefore invalidates
        # every prior frame handle and removes orphaned files from this dedicated root.
        for path in _FRAME_ROOT.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
        _STORE_READY = True


def _delete_frame(frame: FrameRecord):
    _FRAMES.pop(frame.frame_id, None)
    try:
        _media_client().call("release", owner=_MEDIA_OWNER, media_id=frame.media_id)
    except (OSError, MediaClientError):
        # The shared store TTL remains the outer cleanup boundary if the service
        # is unavailable while a local compatibility handle is released.
        pass


def _require_frame(frame_id: str) -> FrameRecord:
    if not _FRAME_ID.fullmatch(frame_id):
        raise SchemaValidationError("frame_id格式无效", capability="camera")
    with _LOCK:
        _cleanup()
        frame = _FRAMES.get(frame_id)
        if frame is None:
            raise ResourceNotFoundError(
                f"摄像头帧不存在或已经释放: {frame_id}", capability="camera")
        return frame


def resolve_frame(frame_id: str) -> Path:
    """Internal-only resolver for trusted local media consumers."""
    frame = _require_frame(frame_id)
    try:
        result = _media_client().call(
            "resolve", owner=_MEDIA_OWNER, media_id=frame.media_id)
        return Path(result["path"])
    except (OSError, KeyError, MediaClientError) as exc:
        raise ResourceNotFoundError(
            "摄像头帧媒体不存在或已经过期", capability="camera") from exc


def snapshot(args):
    camera = _camera_from_args(args)
    formats = _run_format_probe(camera["device"])
    source = _select_source(formats, args.get("source") or {})
    filters, width, height, transform = _validate_transform(source, args.get("transform") or {})
    quality = args.get("quality", 86)
    frame_id = f"frame_{secrets.token_urlsafe(18)}"
    _ensure_store()
    temporary = _FRAME_ROOT / f".{frame_id}.tmp.jpg"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "v4l2", "-input_format", source["pixel_format"],
        "-video_size", f"{source['width']}x{source['height']}",
        "-framerate", str(source["fps"]), "-i", camera["device"],
        "-frames:v", "1",
    ]
    if filters:
        command += ["-vf", ",".join(filters)]
    # FFmpeg's MJPEG qscale is inverse (2 is high quality, 20 is low).
    qscale = max(2, min(20, round(22 - quality / 5)))
    command += ["-q:v", str(qscale), "-update", "1", "-y", str(temporary)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=12, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        temporary.unlink(missing_ok=True)
        raise HardwareError(
            "摄像头截图失败", capability="camera.snapshot",
            details={"reason": type(exc).__name__}) from exc
    if result.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        message = (result.stderr or "ffmpeg failed").strip()[-500:]
        raise HardwareError(
            "摄像头截图失败", capability="camera.snapshot", details={"ffmpeg": message})
    size = temporary.stat().st_size
    if size <= 0 or size > _MAX_FRAME_BYTES:
        temporary.unlink(missing_ok=True)
        raise HardwareError("摄像头帧大小异常", capability="camera.snapshot", details={"size": size})
    os.chmod(temporary, 0o600)
    try:
        media = _media_client().create_from_path(
            _MEDIA_OWNER, "image/jpeg", temporary, ttl_seconds=_FRAME_TTL_SECONDS)
    except (OSError, MediaClientError) as exc:
        raise HardwareError(
            "共享Media Service无法接收摄像头帧", capability="camera.snapshot",
            details={"reason": type(exc).__name__}) from exc
    finally:
        temporary.unlink(missing_ok=True)
    now = time.time()
    frame = FrameRecord(
        frame_id=frame_id, media_id=media["media_id"],
        camera_id=camera["camera_id"], mime_type="image/jpeg",
        width=width, height=height, source=source, transform=transform,
        size=media["size"], created_at=now, expires_at=media["expires_at"])
    with _LOCK:
        _cleanup(now)
        while len(_FRAMES) >= _MAX_FRAMES:
            _delete_frame(min(_FRAMES.values(), key=lambda item: item.created_at))
        _FRAMES[frame_id] = frame
    return frame.public()


def frame_info(args):
    return _require_frame(args["frame_id"]).public()


def release_frame(args):
    frame = _require_frame(args["frame_id"])
    with _LOCK:
        _delete_frame(frame)
    return {"frame_id": frame.frame_id, "released": True}


ACTIONS = {
    "camera.list": list_cameras,
    "camera.get_info": get_info,
    "camera.snapshot": snapshot,
    "camera.frame_info": frame_info,
    "camera.release_frame": release_frame,
}
