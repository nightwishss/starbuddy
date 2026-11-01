"""Owner-only Unix-socket service for unified semantic perception state."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socketserver
import stat
from pathlib import Path

from .client import DEFAULT_SOCKET
from .context import ContextError, SemanticContext
from .providers import ProviderPoller


LOG = logging.getLogger("vq.perception-context")
MAX_REQUEST = 32 * 1024


class ContextService:
    def __init__(self, context, poller=None, fusion=None):
        self.context = context
        self.poller = poller
        self.fusion = fusion

    def _snapshot(self):
        # The context and core locks protect only bounded in-memory work.
        # Never take FusionHost.lock here: its poll tick also does I/O. Using
        # the same semantic view for ingestion and response prevents a fresh
        # observation being rejected by an older fusion view until next poll.
        with self.context.lock:
            snapshot = self.context.snapshot()
            if self.fusion is not None:
                with self.fusion.core.lock:
                    if not self.fusion.closed:
                        self.fusion._ingest_semantic(snapshot)
                    snapshot["fusion"] = self.fusion.snapshot()
            return snapshot

    def dispatch(self, request):
        if not isinstance(request, dict):
            raise ContextError("request must be an object")
        operation = request.get("operation")
        if not isinstance(operation, str):
            raise ContextError("operation must be a string")
        values = {key: value for key, value in request.items() if key != "operation"}
        if operation == "context.snapshot" and not values:
            return self._snapshot()
        if operation == "context.observe" and set(values) == {"observation"}:
            if not isinstance(values["observation"], dict):
                raise ContextError("observation must be an object")
            with self.context.lock:
                result = self.context.observe(**values["observation"])
                if result["accepted"] and self.fusion is not None:
                    self._snapshot()
                return result
        if operation == "context.stats" and not values:
            value = self.context.stats()
            value["providers"] = self.poller.status() if self.poller is not None else {}
            if self.fusion is not None:
                value["fusion"] = {**self.fusion.core.stats(), **self.fusion.stats()}
            return value
        if self.fusion is not None and operation.startswith("fusion."):
            # Socket ownership is the existing same-UID host trust boundary.
            # Clients cannot grant themselves identity access or select a
            # previous user's private session through this API.
            session = values.get("session_id")
            if (operation in {"fusion.snapshot", "fusion.explain"}
                    and session is not None and session != self.fusion.active_session):
                raise ContextError("session is not the active interaction")
            if operation == "fusion.snapshot" and set(values) <= {"session_id"}:
                return self.fusion.snapshot(**values)
            if operation == "fusion.explain" and set(values) <= {"evidence_id", "session_id"}:
                return self.fusion.explain(**values)
            if operation == "fusion.snooze" and set(values) <= {"session_id", "seconds", "origin", "echo"}:
                return self.fusion.snooze(**values)
            if operation == "fusion.posture.proposal" and not values:
                return self.fusion.issue_guard()
            if operation == "fusion.interaction" and not values:
                return self.fusion.interaction.read()
            if operation == "fusion.greeting.proposal" and set(values) == {"event"}:
                return self.fusion.greetings.propose(values["event"])
            if operation == "fusion.validate" and set(values) == {"guard", "job_id"}:
                return self.fusion.authorize_guard(**values)
        raise ContextError("invalid operation")


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        raw = self.rfile.readline(MAX_REQUEST + 1)
        if not raw or len(raw) > MAX_REQUEST:
            return self._write({"ok": False, "error": "invalid request size"})
        try:
            request = json.loads(raw)
            result = self.server.service.dispatch(request)
            self._write({"ok": True, "result": result})
        except (ContextError, TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._write({"ok": False, "error": str(exc)[:160]})
        except Exception:
            LOG.exception("context request failed")
            self._write({"ok": False, "error": "context service error"})

    def _write(self, value):
        self.wfile.write((json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode())


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    args = parser.parse_args()
    if not 0.5 <= args.poll_interval <= 10:
        parser.error("poll interval must be 0.5..10 seconds")
    logging.basicConfig(level=logging.INFO)
    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        if not stat.S_ISSOCK(socket_path.stat().st_mode):
            raise RuntimeError("refusing to replace non-socket context path")
        socket_path.unlink()
    context = SemanticContext()
    from interaction_context.core import FusionCore
    from interaction_context.host import FusionHost
    fusion = FusionHost(FusionCore())
    poller = ProviderPoller(context, interval=args.poll_interval, fusion=fusion)
    def semantic_snapshot():
        return {**context.snapshot(), "providers": poller.status()}
    fusion.semantic_provider = semantic_snapshot
    server = Server(str(socket_path), Handler)
    server.service = ContextService(context, poller, fusion)
    os.chmod(socket_path, 0o600)
    poller.start()
    try:
        server.serve_forever()
    finally:
        server.server_close()
        poller.close()
        fusion.close()
        socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
