"""Owner-only Unix Socket service for persistent local task orchestration."""

from __future__ import annotations

import argparse
import json
import logging
import os
import socketserver
from pathlib import Path

from .engine import TaskOrchestrator, TaskOrchestratorError


LOG = logging.getLogger("vq.task-orchestrator")
MAX_REQUEST = 32 * 1024


class TaskService:
    def __init__(self, orchestrator, *, default_timezone="UTC"):
        self.orchestrator = orchestrator
        self.default_timezone = default_timezone

    def dispatch(self, request: dict):
        if not isinstance(request, dict):
            raise TaskOrchestratorError("invalid_request", "request must be an object")
        operation = request.get("operation")
        values = {key: value for key, value in request.items() if key != "operation"}
        if operation == "task.create" and set(values) <= {
            "utterance", "timezone", "source_conversation_id",
            "proposal", "request_id",
        }:
            return self.orchestrator.create(
                values.get("utterance"), timezone=values.get("timezone", self.default_timezone),
                source_conversation_id=values.get("source_conversation_id"),
                proposal=values.get("proposal"),
                request_id=values.get("request_id"),
            )
        if operation == "task.get" and set(values) == {"task_id"}:
            return self.orchestrator.get(values["task_id"])
        if operation == "task.list" and set(values) <= {"states", "limit"}:
            return self.orchestrator.list(states=values.get("states"), limit=values.get("limit", 64))
        if operation == "task.cancel" and set(values) == {"task_id"}:
            return self.orchestrator.cancel(values["task_id"])
        if operation == "task.stats" and not values:
            return self.orchestrator.stats()
        raise TaskOrchestratorError("invalid_operation", "invalid task operation or arguments")


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        raw = self.rfile.readline(MAX_REQUEST + 1)
        if not raw or len(raw) > MAX_REQUEST:
            return self._write({"ok": False, "code": "invalid_request", "error": "invalid request size"})
        try:
            request = json.loads(raw)
            result = self.server.service.dispatch(request)
            self._write({"ok": True, "result": result})
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._write({"ok": False, "code": "invalid_json", "error": "invalid JSON"})
        except TaskOrchestratorError as exc:
            self._write({"ok": False, "code": exc.code, "error": str(exc)})
        except Exception:
            LOG.exception("task request failed")
            self._write({"ok": False, "code": "service_error", "error": "task service error"})

    def _write(self, value):
        self.wfile.write((json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode())


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", default=os.environ.get(
        "VQ_TASK_ORCHESTRATOR_SOCKET", f"/run/user/{os.getuid()}/vq-task-orchestrator.sock"))
    parser.add_argument("--state", default=os.environ.get(
        "VQ_TASK_STATE", str(Path.home() / ".local/state/vq-platform/tasks.json")))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    from event_service import EventClient
    from gateway.llm import LocalLLMClient
    from runtime.client import RuntimeClient
    from .planner import TaskPlanner
    from .skills import RuntimeSkillExecutor
    from .store import TaskStore

    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    runtime = RuntimeClient(timeout=35)
    orchestrator = TaskOrchestrator(
        TaskPlanner(LocalLLMClient(timeout=45)), TaskStore(Path(args.state)),
        runtime, EventClient(timeout=2), skills=RuntimeSkillExecutor(runtime),
    )
    orchestrator.start()
    server = Server(str(socket_path), Handler)
    server.service = TaskService(
        orchestrator, default_timezone=os.environ.get("VQ_TASK_TIMEZONE", "UTC"),
    )
    os.chmod(socket_path, 0o600)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        orchestrator.close()
        socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
