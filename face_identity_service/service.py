"""Loopback-only enrollment UI and identity status API."""

import argparse
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .engine import IdentityError, IdentityService, SFaceEngine

STATIC_ROOT = Path(__file__).with_name("static")
DEFAULT_DETECTOR = Path.home() / ".local/share/vq-ai/models/face/yunet/face_detection_yunet_2026may.onnx"
DEFAULT_RECOGNIZER = Path.home() / ".local/share/vq-ai/models/face/sface/face_recognition_sface_2021dec.onnx"
MAX_REQUEST = 16 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "VQIdentity/0.1"
    def log_message(self, *_args): pass

    def do_GET(self):
        if not self._trusted_host(): return self._json(HTTPStatus.FORBIDDEN, {"error": "untrusted_host"})
        path = urlparse(self.path).path
        if path in {"/", "/enroll"}: return self._static("index.html", "text/html; charset=utf-8")
        if path == "/app.js": return self._static("app.js", "text/javascript; charset=utf-8")
        if path == "/style.css": return self._static("style.css", "text/css; charset=utf-8")
        if path == "/status": return self._json(HTTPStatus.OK, self.server.identity.status())
        if path == "/api/identities":
            try: return self._json(HTTPStatus.OK, self.server.identity.identities())
            except Exception as exc:
                return self._json(HTTPStatus.SERVICE_UNAVAILABLE,
                                  {"error": "vault_unavailable", "detail": type(exc).__name__})
        if path == "/frame.jpg":
            try: sequence, frame = self.server.identity.debug_frame()
            except IdentityError as exc: return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": exc.code})
            self.send_response(HTTPStatus.OK); self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame))); self.send_header("X-VQ-Sequence", str(sequence))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); return self.wfile.write(frame)
        return self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self):
        if not self._trusted_host(): return self._json(HTTPStatus.FORBIDDEN, {"error": "untrusted_host"})
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).hostname not in {"127.0.0.1", "localhost", "::1"}:
            return self._json(HTTPStatus.FORBIDDEN, {"error": "untrusted_origin"})
        if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
            return self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "json_required"})
        routes = {"/api/enrollment/start": "start_enrollment", "/api/enrollment/commit": "commit_enrollment",
                  "/api/enrollment/cancel": "cancel_enrollment", "/api/identity/delete": "delete_identity"}
        method = routes.get(urlparse(self.path).path)
        if method is None: return self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_REQUEST: raise IdentityError("invalid_request", "invalid request size")
            values = json.loads(self.rfile.read(length))
            if not isinstance(values, dict): raise IdentityError("invalid_request", "request must be an object")
            result = getattr(self.server.identity, method)(**values)
            return self._json(HTTPStatus.OK, result)
        except IdentityError as exc:
            return self._json(HTTPStatus.BAD_REQUEST, {"error": exc.code, "message": str(exc)})
        except (TypeError, ValueError, json.JSONDecodeError):
            return self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_arguments"})

    def _trusted_host(self):
        host = self.headers.get("Host", "").rsplit(":", 1)[0].strip("[]")
        return host in {"127.0.0.1", "localhost", "::1"}

    def _json(self, status, value):
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)

    def _static(self, name, content_type):
        try: body = (STATIC_ROOT / name).read_bytes()
        except OSError: return self._json(HTTPStatus.NOT_FOUND, {"error": "asset_not_found"})
        self.send_response(HTTPStatus.OK); self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers(); self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8795); parser.add_argument("--detector", default=str(DEFAULT_DETECTOR))
    parser.add_argument("--recognizer", default=str(DEFAULT_RECOGNIZER)); parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=180); args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1"}: parser.error("identity service is loopback only")
    if not 0.5 <= args.fps <= 5: parser.error("fps must be between 0.5 and 5")
    for value in (args.detector, args.recognizer):
        if not Path(value).is_file(): parser.error(f"model not found: {value}")
    identity = IdentityService(SFaceEngine(args.detector, args.recognizer, rotation=args.rotation), fps=args.fps)
    identity.start(); server = ThreadingHTTPServer((args.host, args.port), Handler); server.daemon_threads = True
    server.identity = identity
    try: server.serve_forever()
    finally: identity.close()


if __name__ == "__main__": main()
