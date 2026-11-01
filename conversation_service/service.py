"""Owner-only Unix-socket JSONL server for Conversation Session state."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

from .client import DEFAULT_SOCKET
from .store import ConversationError, ConversationStore


LOG = logging.getLogger("vq.conversation")
MAX_REQUEST = 64 * 1024


class ConversationService:
    def __init__(self, store: ConversationStore | None = None):
        self.store = store or ConversationStore()

    async def dispatch(self, request: dict) -> dict:
        if not isinstance(request, dict):
            raise ConversationError("invalid_request", "request must be an object")
        operation = request.get("operation")
        values = {key: value for key, value in request.items() if key != "operation"}
        if operation == "ensure":
            return self.store.ensure(**values)
        if operation == "context":
            return self.store.context(**values)
        if operation == "turn.start":
            return self.store.start_turn(**values)
        if operation == "turn.complete":
            return self.store.complete_turn(**values)
        if operation == "turn.fail":
            return self.store.end_turn(status="failed", **values)
        if operation == "turn.cancel":
            return self.store.end_turn(status="cancelled", **values)
        if operation == "memory.set":
            return self.store.remember(**values)
        if operation == "memory.forget":
            return self.store.forget(**values)
        if operation == "reset":
            return self.store.reset(**values)
        if operation == "stats":
            if values:
                raise ConversationError("invalid_arguments", "stats takes no arguments")
            return self.store.stats()
        if operation == "wait":
            return await asyncio.to_thread(self.store.wait, **values)
        raise ConversationError("unknown_operation", "unknown operation")

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            raw = await reader.readline()
            if not raw or len(raw) > MAX_REQUEST or not raw.endswith(b"\n"):
                raise ConversationError("invalid_request", "invalid request framing")
            try:
                request = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ConversationError("invalid_json", "request is not valid JSON") from exc
            result = await self.dispatch(request)
            response = {"ok": True, "result": result}
        except ConversationError as exc:
            response = {"ok": False, "code": exc.code, "error": str(exc)}
        except (TypeError, ValueError) as exc:
            response = {"ok": False, "code": "invalid_arguments", "error": str(exc)}
        except Exception as exc:
            LOG.exception("conversation request failed")
            response = {"ok": False, "code": "internal_error", "error": type(exc).__name__}
        writer.write((json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()


async def run(socket_path: str, store: ConversationStore | None = None) -> None:
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    service = ConversationService(store)
    server = await asyncio.start_unix_server(service.handle, path=socket_path)
    os.chmod(socket_path, 0o600)
    LOG.info("conversation service ready socket=%s persistent=false", socket_path)
    try:
        async with server:
            await server.serve_forever()
    finally:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    asyncio.run(run(args.socket))


if __name__ == "__main__":
    main()
