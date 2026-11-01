"""Owner-only Unix socket for the shared ephemeral media data plane."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import stat
from pathlib import Path

from .client import default_socket_path
from .store import MediaError, SharedMediaStore

MAX_REQUEST = 64 * 1024


def default_root() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return Path(runtime) / "vq-media"


class MediaServer:
    def __init__(self, root=None, socket_path=None):
        self.store = SharedMediaStore(Path(root or default_root()))
        self.socket_path = Path(socket_path or default_socket_path())
        self.server = None

    async def start(self):
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            if not stat.S_ISSOCK(self.socket_path.stat().st_mode):
                raise RuntimeError("拒绝覆盖非socket路径")
            self.socket_path.unlink()
        self.server = await asyncio.start_unix_server(self._handle, path=str(self.socket_path), limit=MAX_REQUEST + 1)
        os.chmod(self.socket_path, 0o600)

    async def close(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.socket_path.unlink(missing_ok=True)

    async def _handle(self, reader, writer):
        try:
            raw = await reader.readline()
            if not raw or len(raw) > MAX_REQUEST:
                response = self._error("PROTOCOL_ERROR", "Media Service请求无效")
            else:
                try:
                    request = json.loads(raw)
                    response = {"ok": True, "result": self.dispatch(request)}
                except MediaError as exc:
                    response = self._error(exc.code, str(exc))
                except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                    response = self._error("PROTOCOL_ERROR", "Media Service请求格式无效")
            writer.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    def dispatch(self, request: dict):
        if not isinstance(request, dict) or not isinstance(request.get("op"), str):
            raise MediaError("PROTOCOL_ERROR", "Media Service请求格式无效")
        op = request["op"]
        if op == "begin":
            return self.store.begin(request["owner"], request["mime_type"],
                                    kind=request.get("kind"), ttl_seconds=request.get("ttl_seconds"))
        if op == "commit":
            return self.store.commit(request["upload_id"], request["owner"]).public()
        if op == "abort":
            self.store.abort(request["upload_id"], request["owner"])
            return {"aborted": True}
        if op == "info":
            return self.store.get(request["media_id"], request["owner"]).public()
        if op == "resolve":
            return {"media_id": request["media_id"],
                    "path": str(self.store.resolve(request["media_id"], request["owner"]))}
        if op == "release":
            self.store.release(request["media_id"], request["owner"])
            return {"media_id": request["media_id"], "released": True}
        if op == "list":
            values = [item.public() for item in self.store.list_owned(request["owner"])]
            return {"media": values, "count": len(values)}
        if op == "cleanup":
            return {"removed": self.store.cleanup()}
        raise MediaError("UNKNOWN_OPERATION", "未知Media Service操作")

    @staticmethod
    def _error(code, message):
        return {"ok": False, "error": {"code": code, "message": message}}


async def async_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--socket", type=Path, default=default_socket_path())
    args = parser.parse_args()
    server = MediaServer(args.root, args.socket)
    await server.start()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    try:
        await stop.wait()
    finally:
        await server.close()


if __name__ == "__main__":
    asyncio.run(async_main())
