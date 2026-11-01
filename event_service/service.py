"""Owner-only Unix-socket service for bounded semantic events."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from .client import DEFAULT_SOCKET
from .store import EventError, EventStore


LOG = logging.getLogger("vq.events")
MAX_REQUEST = 32 * 1024


class EventService:
    def __init__(self, store=None):
        self.store = store or EventStore()

    async def dispatch(self, request):
        if not isinstance(request, dict):
            raise EventError("invalid_request", "request must be an object")
        operation = request.get("operation")
        values = {k: v for k, v in request.items() if k != "operation"}
        if operation == "publish":
            return self.store.publish(**values)
        if operation == "read":
            return self.store.read(**values)
        if operation == "wait":
            return await asyncio.to_thread(self.store.wait, **values)
        if operation == "stats" and not values:
            return self.store.stats()
        raise EventError("unknown_operation", "unknown operation")

    async def handle(self, reader, writer):
        try:
            raw = await reader.readline()
            if not raw or len(raw) > MAX_REQUEST or not raw.endswith(b"\n"):
                raise EventError("invalid_request", "invalid request framing")
            request = json.loads(raw)
            response = {"ok": True, "result": await self.dispatch(request)}
        except EventError as exc:
            response = {"ok": False, "code": exc.code, "error": str(exc)}
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as exc:
            response = {"ok": False, "code": "invalid_arguments", "error": str(exc)}
        except Exception as exc:
            LOG.exception("event request failed")
            response = {"ok": False, "code": "internal_error", "error": type(exc).__name__}
        writer.write((json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()


async def run(socket_path=DEFAULT_SOCKET):
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    server = await asyncio.start_unix_server(EventService().handle, path=socket_path)
    os.chmod(socket_path, 0o600)
    try:
        async with server:
            await server.serve_forever()
    finally:
        path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(args.socket))


if __name__ == "__main__":
    main()
