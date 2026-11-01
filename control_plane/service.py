"""Owner-only model registry and resource scheduler service."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .client import DEFAULT_SOCKET
from .registry import ModelRegistry
from .scheduler import ResourceScheduler, SchedulerError


class ControlService:
    def __init__(self, registry=None, scheduler=None):
        self.registry = registry or ModelRegistry()
        self.scheduler = scheduler or ResourceScheduler(
            state_path=Path(f"/run/user/{os.getuid()}/vq-llm-lane.json"),
            foreground_npu=os.environ.get("VQ_LLM_NPU_ADMISSION", "0") == "1")

    async def dispatch(self, request):
        if not isinstance(request, dict):
            raise SchedulerError("invalid_request", "request must be an object")
        op = request.get("operation")
        values = {k: v for k, v in request.items() if k != "operation"}
        if op == "models.list": return self.registry.list(**values)
        if op == "models.get": return self.registry.get(**values)
        if op == "models.acquire":
            model = self.registry.get(model_id=values.pop("model_id"))
            return self.scheduler.acquire(
                resources=model["resources"], purpose=model["task"], **values,
            )
        if op == "resources.acquire": return self.scheduler.acquire(**values)
        if op == "resources.renew": return self.scheduler.renew(**values)
        if op == "resources.release": return self.scheduler.release(**values)
        if op == "resources.status" and not values: return self.scheduler.status()
        if op == "llm.enter": return self.scheduler.llm_enter(**values)
        if op == "llm.finish": return self.scheduler.llm_finish(**values)
        if op == "llm.status" and not values: return self.scheduler.llm_status()
        raise SchedulerError("unknown_operation", "unknown operation")

    async def handle(self, reader, writer):
        try:
            raw = await reader.readline()
            if not raw or len(raw) > 32 * 1024 or not raw.endswith(b"\n"):
                raise SchedulerError("invalid_request", "invalid framing")
            result = await self.dispatch(json.loads(raw))
            value = {"ok": True, "result": result}
        except SchedulerError as exc:
            value = {"ok": False, "code": exc.code, "error": str(exc), "details": exc.details}
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            value = {"ok": False, "code": "invalid_arguments", "error": str(exc)}
        except Exception as exc:
            value = {"ok": False, "code": "internal_error", "error": type(exc).__name__}
        writer.write((json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
        await writer.drain(); writer.close(); await writer.wait_closed()


async def run(socket_path=DEFAULT_SOCKET):
    path = Path(socket_path); path.parent.mkdir(parents=True, exist_ok=True); path.unlink(missing_ok=True)
    server = await asyncio.start_unix_server(ControlService().handle, path=socket_path)
    os.chmod(socket_path, 0o600)
    try:
        async with server: await server.serve_forever()
    finally:
        path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--socket", default=DEFAULT_SOCKET)
    args = parser.parse_args(); asyncio.run(run(args.socket))


if __name__ == "__main__": main()
