"""Memory-only sampler and private JSONL Unix-socket telemetry API."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import stat
import threading
from collections import deque
from pathlib import Path

from .client import default_socket_path
from .collector import TelemetryCollector


MAX_REQUEST = 4096


class TelemetryStore:
    def __init__(self, maximum=720):
        self.values = deque(maxlen=maximum)
        self.lock = threading.Lock()

    def append(self, value):
        with self.lock:
            self.values.append(value)

    def latest(self):
        with self.lock:
            return self.values[-1] if self.values else None

    def history(self, limit):
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 360:
            raise ValueError("history limit must be 1..360")
        with self.lock:
            return list(self.values)[-limit:]


class TelemetryServer:
    def __init__(self, *, socket_path=None, interval=5.0, collector=None, store=None):
        self.socket_path = Path(socket_path or default_socket_path())
        self.interval = interval
        self.collector = collector or TelemetryCollector()
        self.store = store or TelemetryStore()
        self.server = None
        self.sampler = None
        self.stop_event = asyncio.Event()

    async def start(self):
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            if not stat.S_ISSOCK(self.socket_path.stat().st_mode):
                raise RuntimeError("refusing to replace non-socket telemetry path")
            self.socket_path.unlink()
        self.server = await asyncio.start_unix_server(
            self._handle, path=str(self.socket_path), limit=MAX_REQUEST + 1
        )
        os.chmod(self.socket_path, 0o600)
        self.sampler = asyncio.create_task(self._sample_loop())

    async def close(self):
        self.stop_event.set()
        if self.sampler:
            await self.sampler
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        self.socket_path.unlink(missing_ok=True)

    async def _sample_loop(self):
        while not self.stop_event.is_set():
            started = asyncio.get_running_loop().time()
            try:
                value = await asyncio.to_thread(self.collector.sample)
                self.store.append(value)
            except Exception:
                pass
            remaining = max(0.0, self.interval - (asyncio.get_running_loop().time() - started))
            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                pass

    async def _handle(self, reader, writer):
        try:
            raw = await reader.readline()
            if not raw or len(raw) > MAX_REQUEST:
                response = {"ok": False, "error": {"message": "invalid request"}}
            else:
                try:
                    request = json.loads(raw)
                    op = request.get("op")
                    if op == "latest":
                        result = self.store.latest()
                    elif op == "history":
                        result = {"samples": self.store.history(request.get("limit", 120))}
                    else:
                        raise ValueError("unknown operation")
                    if result is None:
                        raise ValueError("no telemetry sample available")
                    response = {"ok": True, "result": result}
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    response = {"ok": False, "error": {"message": str(exc)}}
            writer.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode() + b"\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()


async def async_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, default=default_socket_path())
    parser.add_argument("--interval", type=float, default=5.0)
    arguments = parser.parse_args()
    if not 1 <= arguments.interval <= 60:
        parser.error("interval must be 1..60 seconds")
    server = TelemetryServer(socket_path=arguments.socket, interval=arguments.interval)
    await server.start()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, server.stop_event.set)
    try:
        await server.stop_event.wait()
    finally:
        await server.close()


if __name__ == "__main__":
    asyncio.run(async_main())
