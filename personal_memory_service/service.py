"""Owner-only Unix-socket service for explicit product memory."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from .client import DEFAULT_SOCKET
from .store import PersonalMemoryError, PersonalMemoryStore

MAX_REQUEST = 64 * 1024
OPERATIONS = {
    "profile.get": "profile_get", "profile.set": "profile_set", "profile.forget": "profile_forget",
    "memory.put": "memory_put", "memory.recall": "memory_recall", "memory.forget": "memory_forget",
    "profile.reset": "reset", "stats": "stats",
}


async def handle(store, reader, writer):
    try:
        raw = await reader.readline()
        if not raw or len(raw) > MAX_REQUEST or not raw.endswith(b"\n"):
            raise PersonalMemoryError("invalid_request", "invalid request framing")
        request = json.loads(raw)
        operation = request.pop("operation", None)
        method = OPERATIONS.get(operation)
        if method is None:
            raise PersonalMemoryError("unknown_operation", "unknown operation")
        result = await asyncio.to_thread(getattr(store, method), **request)
        response = {"ok": True, "result": result}
    except PersonalMemoryError as exc:
        response = {"ok": False, "code": exc.code, "error": str(exc)}
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        response = {"ok": False, "code": "invalid_arguments", "error": type(exc).__name__}
    except Exception as exc:
        response = {"ok": False, "code": "internal_error", "error": type(exc).__name__}
    writer.write((json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
    await writer.drain(); writer.close(); await writer.wait_closed()


async def run(socket_path, database_path=None):
    path = Path(socket_path); path.parent.mkdir(parents=True, exist_ok=True); path.unlink(missing_ok=True)
    store = PersonalMemoryStore(database_path)
    server = await asyncio.start_unix_server(lambda r, w: handle(store, r, w), path=socket_path)
    os.chmod(path, 0o600)
    try:
        async with server: await server.serve_forever()
    finally:
        path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--socket", default=DEFAULT_SOCKET)
    parser.add_argument("--database"); args = parser.parse_args()
    asyncio.run(run(args.socket, args.database))


if __name__ == "__main__": main()

