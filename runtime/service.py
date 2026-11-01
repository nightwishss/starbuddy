"""VQ Capability Runtime本地Unix Socket服务。"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import stat
import struct
import time
import uuid
from pathlib import Path

from runtime.catalog import build_registry
from runtime.dispatcher import execute_capability
from runtime.errors import (
    CapabilityError,
    ImplementationError,
    PermissionDeniedError,
    SchemaValidationError,
    TimeoutError as CapabilityTimeoutError,
)
from runtime.policy import AccessPolicy
from runtime.resources import ResourceTracker

MAX_REQUEST_BYTES = 1024 * 1024
MAX_TIMEOUT_MS = 30_000


def default_socket_path():
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    return Path(runtime_dir) / "vq-runtime.sock"


class RuntimeEngine:
    """Serial hardware lane plus a bounded speech lane for model execution."""

    def __init__(self, registry=None, policy=None):
        self.registry = registry or build_registry()
        self.policy = policy or AccessPolicy()
        self.resources = ResourceTracker()
        self._lock = asyncio.Lock()
        self._speech_lock = asyncio.Lock()
        self._audio_output_lock = asyncio.Lock()
        self._closing = False

    async def handle(self, request, peer_uid):
        request_id = request.get("id") if isinstance(request, dict) else None
        try:
            request = self._validate_request(request)
            request_id = request["id"]
            self.policy.authorize_peer(peer_uid)
            entry = self.registry.get(request["capability"])
            self.policy.authorize_capability(request["capability"], entry["metadata"])

            timeout_ms = min(
                request.get("timeout_ms") or entry["metadata"].get("timeout_ms", MAX_TIMEOUT_MS),
                MAX_TIMEOUT_MS,
            )
            started = time.monotonic()
            # Keep the lane owned until the native worker actually finishes,
            # including when a socket handler is cancelled during shutdown.
            return await asyncio.shield(self._execute(request, peer_uid, timeout_ms, started))
        except CapabilityError as exc:
            return {"id": request_id, "ok": False, "error": exc.to_dict()}
        except Exception as exc:
            error = ImplementationError(f"Runtime内部错误: {exc!r}")
            return {"id": request_id, "ok": False, "error": error.to_dict()}

    async def _execute(self, request, peer_uid, timeout_ms, started):
        # Speech has separate model locks and media-owner authorization. All
        # other module-level device/resource tables retain their serial lane.
        if request["capability"] in {"speech.synthesize", "speech.transcribe"}:
            lock = self._speech_lock
        elif request["capability"] in {
            "audio.start_stream_playback", "audio.append_stream_playback",
            "audio.finish_stream_playback", "audio.playback_status", "audio.stop_playback",
            "speech.status",
        }:
            lock = self._audio_output_lock
        else:
            lock = self._lock
        async with lock:
            if self._closing:
                raise ImplementationError("Runtime正在关闭")
            self.resources.authorize(peer_uid, request["args"])
            queue_wait_ms = (time.monotonic() - started) * 1000
            if queue_wait_ms >= timeout_ms:
                raise CapabilityTimeoutError("请求在队列中已过期，未执行",
                                             capability=request["capability"],
                                             details={"queue_wait_ms": round(queue_wait_ms, 2),
                                                      "execute_ms": 0, "timeout_ms": timeout_ms})
            executing = time.monotonic()
            result = await asyncio.to_thread(
                execute_capability, self.registry,
                request["capability"], request["args"])
            execute_ms = (time.monotonic() - executing) * 1000
            elapsed_ms = round(queue_wait_ms + execute_ms, 2)
            new_resource = self.resources.record_result(
                peer_uid, request["capability"], request["args"], result)
            if elapsed_ms > timeout_ms:
                if new_resource:
                    key, handle, record = new_resource
                    try:
                        await asyncio.to_thread(
                            execute_capability, self.registry,
                            record.close_capability,
                            {record.argument_name: handle})
                    finally:
                        self.resources.remove(key, handle)
                raise CapabilityTimeoutError(
                    f"调用完成但超过时限: {elapsed_ms}ms > {timeout_ms}ms",
                    capability=request["capability"],
                    details={"elapsed_ms": elapsed_ms, "timeout_ms": timeout_ms,
                             "queue_wait_ms": round(queue_wait_ms, 2),
                             "execute_ms": round(execute_ms, 2)})
            return {"id": request["id"], "ok": True, "result": result,
                    "meta": {"elapsed_ms": elapsed_ms,
                             "queue_wait_ms": round(queue_wait_ms, 2),
                             "execute_ms": round(execute_ms, 2)}}

    @staticmethod
    def _validate_request(request):
        if not isinstance(request, dict):
            raise SchemaValidationError("请求必须是JSON object")
        allowed = {"id", "capability", "args", "timeout_ms"}
        extra = sorted(set(request) - allowed)
        if extra:
            raise SchemaValidationError(
                f"请求含未知字段: {extra}", details={"fields": extra})
        request_id = request.get("id")
        capability = request.get("capability")
        args = request.get("args", {})
        timeout_ms = request.get("timeout_ms")
        if not isinstance(request_id, str) or not request_id:
            raise SchemaValidationError("id必须是非空字符串")
        if not isinstance(capability, str) or not capability:
            raise SchemaValidationError("capability必须是非空字符串")
        if not isinstance(args, dict):
            raise SchemaValidationError("args必须是object")
        if timeout_ms is not None and (
                isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int)
                or timeout_ms < 1 or timeout_ms > MAX_TIMEOUT_MS):
            raise SchemaValidationError(
                f"timeout_ms必须在1..{MAX_TIMEOUT_MS}之间")
        return {"id": request_id, "capability": capability,
                "args": args, "timeout_ms": timeout_ms}

    async def cleanup(self):
        self._closing = True
        async with self._speech_lock, self._lock, self._audio_output_lock:
            for (key, handle), record in reversed(self.resources.records()):
                try:
                    await asyncio.to_thread(
                        execute_capability, self.registry,
                        record.close_capability, {record.argument_name: handle})
                except Exception:
                    logging.exception("resource_cleanup_failed", extra={"handle": handle})
            self.resources.clear()
            # Model children own native resources beyond individual RPCs.
            from capability.speech.impl import close as close_speech
            await asyncio.to_thread(close_speech)


class RuntimeServer:
    def __init__(self, socket_path=None, engine=None):
        self.socket_path = Path(socket_path or default_socket_path())
        self.engine = engine or RuntimeEngine()
        self.server = None

    async def start(self):
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            mode = self.socket_path.stat().st_mode
            if not stat.S_ISSOCK(mode):
                raise RuntimeError(f"拒绝覆盖非socket路径: {self.socket_path}")
            self.socket_path.unlink()
        self.server = await asyncio.start_unix_server(
            self._handle_client, path=str(self.socket_path),
            limit=MAX_REQUEST_BYTES + 1)
        os.chmod(self.socket_path, 0o600)
        logging.info("runtime_started socket=%s", self.socket_path)

    async def serve_forever(self):
        await self.start()
        async with self.server:
            await self.server.serve_forever()

    async def close(self):
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        await self.engine.cleanup()
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass

    async def _handle_client(self, reader, writer):
        peer_uid = self._peer_uid(writer)
        try:
            while True:
                try:
                    raw = await reader.readline()
                except ValueError:
                    await self._write(writer, self._protocol_error(None, "请求超过1MiB限制"))
                    break
                if not raw:
                    break
                if len(raw) > MAX_REQUEST_BYTES:
                    await self._write(writer, self._protocol_error(None, "请求超过1MiB限制"))
                    break
                try:
                    request = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    await self._write(writer, self._protocol_error(None, f"非法JSON: {exc}"))
                    continue
                response = await self.engine.handle(request, peer_uid)
                self._audit(peer_uid, request, response)
                await self._write(writer, response)
        finally:
            writer.close()
            await writer.wait_closed()

    @staticmethod
    def _peer_uid(writer):
        sock = writer.get_extra_info("socket")
        if hasattr(socket, "SO_PEERCRED"):
            _pid, uid, _gid = struct.unpack(
                "3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            return uid
        return os.getuid()

    @staticmethod
    async def _write(writer, response):
        payload = (json.dumps(response, ensure_ascii=False) + "\n").encode()
        if len(payload) > MAX_REQUEST_BYTES:
            error = ImplementationError("响应超过1MiB限制")
            payload = (json.dumps({
                "id": response.get("id"), "ok": False,
                "error": error.to_dict()}, ensure_ascii=False) + "\n").encode()
        writer.write(payload)
        await writer.drain()

    @staticmethod
    def _audit(peer_uid, request, response):
        event = {
            "event": "capability_call",
            "id": request.get("id"),
            "peer_uid": peer_uid,
            "capability": request.get("capability"),
            "ok": response.get("ok", False),
        }
        if response.get("ok"):
            for name in ("elapsed_ms", "queue_wait_ms", "execute_ms"):
                event[name] = response.get("meta", {}).get(name)
        else:
            event["error_code"] = response.get("error", {}).get("code")
        logging.info(json.dumps(event, ensure_ascii=False, separators=(",", ":")))

    @staticmethod
    def _protocol_error(request_id, message):
        error = SchemaValidationError(message)
        return {"id": request_id, "ok": False, "error": error.to_dict()}


def parse_args():
    parser = argparse.ArgumentParser(description="VQ Capability Runtime Service")
    parser.add_argument("--socket", type=Path, default=default_socket_path())
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


async def async_main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s")
    server = RuntimeServer(args.socket)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    try:
        await server.start()
        async with server.server:
            await stop.wait()
    finally:
        await server.close()


def main():
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
