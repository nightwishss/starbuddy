"""Validated Unix-socket client for the MCU-owned 8x13 LED matrix."""

import json
import os
import socket
import time
from pathlib import Path

from runtime.errors import HardwareError

ROWS, COLUMNS, PIXELS = 8, 13, 104
SOCKET_PATH = Path(os.environ.get(
    "VQ_MATRIX_SOCKET",
    "/run/user/1000/vq-matrix-relay.sock",
))

ICONS = {
    "check": ["0000000000000","0000000000000","0000000000010","0000000001100","0100000110000","0010001100000","0001110000000","0000000000000"],
    "cross": ["0110000000110","0011000001100","0001100011000","0000110110000","0000110110000","0001100011000","0011000001100","0110000000110"],
    "heart": ["0000000000000","0011000110000","0111101111000","0111111111000","0011111110000","0001111100000","0000111000000","0000010000000"],
    "smile": ["0001111110000","0010000001000","0100100010100","0100000000100","0101000000100","0100111100100","0010000001000","0001111110000"],
}


def _call(payload):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(1.5)
            client.connect(str(SOCKET_PATH))
            client.sendall(json.dumps(payload).encode() + b"\n")
            raw = client.makefile("rb").readline(4097)
        response = json.loads(raw)
        if not response.get("ok"):
            raise ValueError(response.get("error", "matrix bridge rejected request"))
        return response["result"]
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        raise HardwareError("LED矩阵桥接不可用", capability="matrix", details={"reason": type(exc).__name__}) from exc


def draw(args):
    pixels = args["pixels"]
    result = _call({"op": "draw", "pixels": pixels})
    return {"drawn": True, "lit_pixels": result["lit_pixels"]}


def icon(args):
    brightness = args.get("brightness", 7)
    pixels = [brightness if bit == "1" else 0 for row in ICONS[args["name"]] for bit in row]
    result = draw({"pixels": pixels})
    return {**result, "icon": args["name"]}


def clear(args=None):
    _call({"op": "draw", "pixels": [0] * PIXELS})
    return {"cleared": True}


def blink(args):
    """Blink one built-in icon for a bounded duration, then select a final state."""
    name = args["name"]
    brightness = args.get("brightness", 7)
    duration_seconds = args.get("duration_seconds", 10)
    interval_ms = args.get("interval_ms", 500)
    end_state = args.get("end_state", "on")
    interval = interval_ms / 1000.0
    deadline = time.monotonic() + duration_seconds
    cycles = 0
    while time.monotonic() < deadline:
        icon({"name": name, "brightness": brightness})
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
        if time.monotonic() >= deadline:
            break
        clear()
        cycles += 1
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
    if end_state == "on":
        icon({"name": name, "brightness": brightness})
    else:
        clear()
    return {"blinked": True, "icon": name, "cycles": cycles, "end_state": end_state}


def status(args=None):
    result = _call({"op": "status"})
    return {"ready": result["ready"], "rows": ROWS, "columns": COLUMNS}


ACTIONS = {
    "matrix.draw": draw,
    "matrix.icon": icon,
    "matrix.blink": blink,
    "matrix.clear": clear,
    "matrix.status": status,
}
