"""Bounded owner-only relay between Runtime and the App container socket."""

import argparse
import os
import socket
import socketserver
from pathlib import Path

MAX_LINE = 8192


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        request = self.request.recv(MAX_LINE + 1)
        if not request.endswith(b"\n") or len(request) > MAX_LINE:
            return
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
            upstream.settimeout(2)
            upstream.connect(self.server.upstream)  # type: ignore[attr-defined]
            upstream.sendall(request)
            response = upstream.recv(MAX_LINE + 1)
        if response.endswith(b"\n") and len(response) <= MAX_LINE:
            self.request.sendall(response)


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", default=f"/run/user/{os.getuid()}/vq-matrix-relay.sock")
    parser.add_argument("--upstream", default="/dev/shm/vq-matrix-bridge-1000.sock")
    args = parser.parse_args()
    path = Path(args.listen)
    path.unlink(missing_ok=True)
    server = Server(str(path), Handler)
    server.upstream = args.upstream
    os.chmod(path, 0o600)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
