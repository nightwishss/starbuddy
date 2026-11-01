"""Owner-only local socket around the Arduino RouterBridge matrix function."""

import json
import os
import socket
import threading
from pathlib import Path

from arduino.app_utils import App, Bridge

SOCKET_PATH = Path("/dev/shm/vq-matrix-bridge-1000.sock")


def respond(connection, value):
    connection.sendall(json.dumps(value, separators=(",", ":")).encode() + b"\n")


def handle(connection):
    try:
        request = json.loads(connection.makefile("rb").readline(8193))
        if request == {"op": "status"}:
            respond(connection, {"ok": True, "result": {"ready": True}})
            return
        pixels = request.get("pixels") if request.get("op") == "draw" else None
        if not isinstance(pixels, list) or len(pixels) != 104 or any(type(v) is not int or not 0 <= v <= 7 for v in pixels):
            respond(connection, {"ok": False, "error": "invalid frame"})
            return
        Bridge.call("draw", bytes(pixels))
        respond(connection, {"ok": True, "result": {"lit_pixels": sum(v > 0 for v in pixels)}})
    except Exception as exc:
        print(f"matrix bridge request failed: {type(exc).__name__}: {exc}", flush=True)
        respond(connection, {"ok": False, "error": "bridge failure"})
    finally:
        connection.close()


def serve():
    SOCKET_PATH.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(SOCKET_PATH))
    os.chmod(SOCKET_PATH, 0o600)
    server.listen(4)
    while True:
        connection, _ = server.accept()
        threading.Thread(target=handle, args=(connection,), daemon=True).start()


threading.Thread(target=serve, daemon=True).start()
App.run()
