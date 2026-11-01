"""Restricted adapter for the localhost Arduino QNN detection runner."""

from __future__ import annotations

import http.client
import json
import math
import os
import re
import secrets
from pathlib import Path

from media_service import MediaClient, MediaClientError
from runtime.errors import HardwareError, ResourceNotFoundError, SchemaValidationError

_MEDIA_ID = re.compile(r"^image_[A-Za-z0-9_-]{20,64}$")
_MEDIA_OWNER = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
_RUNTIME_DIR = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
_MEDIA_ROOT = _RUNTIME_DIR / "vq-phone-gateway/media"
_RUNNER_HOST = "127.0.0.1"
_RUNNER_PORT = 1337
_MAX_RESPONSE_BYTES = 1024 * 1024
def _media_client():
    return MediaClient()


def _image_source(media_id: str, media_owner: str | None = None) -> Path:
    if not _MEDIA_ID.fullmatch(media_id):
        raise SchemaValidationError("media_id格式无效", capability="vision.detect_objects")
    if media_owner is not None:
        if not _MEDIA_OWNER.fullmatch(media_owner):
            raise SchemaValidationError("共享图片所有者无效", capability="vision.detect_objects")
        try:
            value = _media_client().call("resolve", owner=media_owner, media_id=media_id)
            return Path(value["path"])
        except (OSError, KeyError, MediaClientError) as exc:
            raise ResourceNotFoundError(
                "共享图片不存在或已经过期", capability="vision.detect_objects") from exc
    path = _MEDIA_ROOT / f"{media_id}.jpg"
    if path.parent.resolve() != _MEDIA_ROOT.resolve() or not path.is_file():
        raise ResourceNotFoundError("图片不存在或已经过期", capability="vision.detect_objects")
    return path


def _post_image(path: Path) -> dict:
    boundary = f"vq-{secrets.token_hex(12)}"
    image = path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="snapshot.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode() + image + f"\r\n--{boundary}--\r\n".encode()
    connection = http.client.HTTPConnection(_RUNNER_HOST, _RUNNER_PORT, timeout=9)
    try:
        connection.request(
            "POST",
            "/api/image",
            body=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        response = connection.getresponse()
        payload = response.read(_MAX_RESPONSE_BYTES + 1)
    except OSError as exc:
        raise HardwareError("本地视觉模型暂时不可用", capability="vision.detect_objects") from exc
    finally:
        connection.close()
    if response.status != 200 or len(payload) > _MAX_RESPONSE_BYTES:
        raise HardwareError("本地视觉推理失败", capability="vision.detect_objects")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HardwareError("本地视觉模型返回格式无效", capability="vision.detect_objects") from exc
    if not isinstance(value, dict):
        raise HardwareError("本地视觉模型返回格式无效", capability="vision.detect_objects")
    return value


def _finite_number(value, default=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return default
    return float(value)


def detect_objects(args):
    confidence = _finite_number(args.get("confidence", 0.3), 0.3)
    raw = _post_image(_image_source(args["media_id"], args.get("media_owner")))
    result = raw.get("result") if isinstance(raw.get("result"), dict) else {}
    boxes = result.get("bounding_boxes") if isinstance(result.get("bounding_boxes"), list) else []
    detections = []
    for box in boxes[:100]:
        if not isinstance(box, dict):
            continue
        score = _finite_number(box.get("value"))
        label = str(box.get("label", "unknown"))[:100]
        if score < confidence:
            continue
        detections.append({
            "label": label,
            "confidence": score,
            "x": _finite_number(box.get("x")),
            "y": _finite_number(box.get("y")),
            "width": _finite_number(box.get("width")),
            "height": _finite_number(box.get("height")),
        })
    timing = raw.get("timing") if isinstance(raw.get("timing"), dict) else {}
    safe_timing = {
        str(key)[:50]: _finite_number(value)
        for key, value in list(timing.items())[:20]
    }
    return {
        "detections": detections,
        "count": len(detections),
        "model": "YOLOX-Nano-QNN-416",
        "backend": "QNN/HTP",
        "timing": safe_timing,
    }


ACTIONS = {"vision.detect_objects": detect_objects}
