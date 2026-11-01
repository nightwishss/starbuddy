"""Loopback status server for low-rate local facial-expression cues."""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from camera_stream_service import CameraStreamClient
from perception_context_service import PerceptionContextClient

from .engine import AffectWorker, FerPlusModel


LOG = logging.getLogger("vq.affect-cues")
DEFAULT_MODEL = Path.home() / ".local/share/vq-ai/models/affect/emotion-ferplus-8.onnx"


def face_status():
    with urllib.request.urlopen("http://127.0.0.1:8794/status", timeout=.75) as response:
        return json.load(response)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        if self.path != "/status" or self.headers.get("Host") not in {
                "127.0.0.1:8796", "localhost:8796"}:
            self.send_error(404)
            return
        body = json.dumps(self.server.worker.status(), separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--fps", type=float, default=.5)
    args = parser.parse_args()
    if not args.model.is_file() or not .2 <= args.fps <= 1:
        parser.error("model must exist and fps must be between 0.2 and 1")
    admission_mode = os.environ.get("VQ_AFFECT_RESOURCE_ADMISSION", "0")
    if admission_mode not in {"0", "1"}:
        parser.error("VQ_AFFECT_RESOURCE_ADMISSION must be 0 or 1")
    resources = None
    if admission_mode == "1":
        from control_plane.client import ControlPlaneClient
        resources = ControlPlaneClient(timeout=.2)
    worker = AffectWorker(
        FerPlusModel(args.model),
        CameraStreamClient(consumer="affect", source="camera.front", rotation=0, timeout=1.5),
        face_status, PerceptionContextClient(timeout=.75), fps=args.fps,
        resources=resources,
    )
    stop = threading.Event()

    def run():
        while not stop.is_set():
            started = time.monotonic()
            worker.tick()
            if worker.last_error:
                LOG.warning("affect_sample_failed error=%s", worker.last_error)
            stop.wait(max(0, worker.interval - (time.monotonic() - started)))

    thread = threading.Thread(target=run, name="vq-affect-cues", daemon=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8796), Handler)
    server.worker = worker
    thread.start()
    try:
        server.serve_forever()
    finally:
        stop.set()
        server.server_close()
        thread.join(timeout=3)


if __name__ == "__main__":
    main()
