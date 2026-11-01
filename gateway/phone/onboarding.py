#!/usr/bin/env python3
"""Public, read-only HTTP onboarding page for the Phone Gateway CA."""

from __future__ import annotations

import argparse
import html
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


PAGE = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>安装 VENTUNO Q Phone Link</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#07100c;color:#edf8f1}
body{margin:0;min-height:100vh;display:grid;place-items:center;padding:22px;box-sizing:border-box}
main{max-width:520px;background:#101c17;border:1px solid #2d4238;border-radius:24px;padding:28px}
.tag{color:#b8ff58;font-size:12px;letter-spacing:.13em}h1{font-size:27px}p,li{line-height:1.7;color:#b9c9c0}
a{display:block;text-align:center;margin-top:14px;padding:14px;border-radius:13px;background:#b8ff58;color:#102000;text-decoration:none;font-weight:750}
a.secondary{background:#1b2a23;color:#edf8f1;border:1px solid #344b40}.note{font-size:13px}
</style></head><body><main><div class="tag">VENTUNO Q · PHONE LINK</div>
<h1>让手机安全连接板卡</h1><ol><li>下载下面的公开 CA 证书。</li>
<li>在三星“设置 → 安全和隐私 → 更多安全设置 → 从设备存储安装 → CA证书”中安装。</li>
<li>返回这里，打开安全控制台并输入板端一次性配对码。</li></ol>
<a href="/VENTUNO-Q-Local-CA.crt" download>下载 CA 证书</a>
<a class="secondary" href="{console_url}">打开安全控制台</a>
<p class="note">此 HTTP 页面只提供公开证书和说明，不接受密码，也不能访问 Hermes 或板卡能力。</p>
</main></body></html>"""


class OnboardingHandler(BaseHTTPRequestHandler):
    server_version = "VQPhoneOnboarding/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/setup"}:
            host = (self.headers.get("Host") or "ventunoq.local").split(":", 1)[0]
            body = PAGE.replace(
                "{console_url}", html.escape(f"https://{host}:8443", quote=True)
            ).encode()
            self._send(HTTPStatus.OK, body, "text/html; charset=utf-8")
            return
        if path == "/VENTUNO-Q-Local-CA.crt":
            body = self.server.ca_path.read_bytes()  # type: ignore[attr-defined]
            self._send(
                HTTPStatus.OK,
                body,
                "application/x-x509-ca-cert",
                {"Content-Disposition": 'attachment; filename="VENTUNO-Q-Local-CA.crt"'},
            )
            return
        if path == "/healthz":
            self._send(HTTPStatus.OK, b'{"ok":true}', "application/json")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        self.send_error(HTTPStatus.METHOD_NOT_ALLOWED)

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send(self, status: HTTPStatus, body: bytes, content_type: str, headers=None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ca", required=True)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), OnboardingHandler)
    server.daemon_threads = True
    server.ca_path = Path(args.ca)  # type: ignore[attr-defined]
    server.serve_forever()


if __name__ == "__main__":
    main()
