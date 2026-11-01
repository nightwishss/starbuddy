"""Bounded, cached readiness probes for the deployed StarBuddy product path."""

from __future__ import annotations

import concurrent.futures
import json
import socket
import threading
import time
import urllib.error
import urllib.request

from media_service import MediaClient
from conversation_service import ConversationClient
from sdk import DeviceClient, SpeechClient


class ProductReadiness:
    def __init__(self, *, runtime=None, media=None, telemetry=None, conversation=None,
                 probes=None, cache_seconds=2.0, clock=time.monotonic):
        if probes is None:
            if runtime is None:
                from runtime.client import RuntimeClient
                runtime = RuntimeClient(timeout=1.5)
            media = media or MediaClient(timeout=1.0)
            conversation = conversation or ConversationClient(timeout=1.0)
            if telemetry is None:
                from telemetry_service import TelemetryClient
                telemetry = TelemetryClient(timeout=1.0)
        self.runtime = runtime
        self.device = DeviceClient(runtime) if runtime is not None else None
        self.speech = SpeechClient(runtime) if runtime is not None else None
        self.media = media
        self.conversation = conversation
        self.clock = clock
        self.cache_seconds = cache_seconds
        self._probes = probes or {
            "runtime": self._runtime,
            "media": self._media,
            "conversation_session": self._conversation,
            "local_llm": lambda: self._http_json(
                "http://127.0.0.1:9001/v1/models", self._valid_models),
            "local_agent": lambda: self._http_json(
                "http://127.0.0.1:9003/healthz", lambda value: value.get("ok") is True),
            "vision_runner": self._vision,
            "display": lambda: self._http_json(
                "http://127.0.0.1:8765/healthz", lambda value: value.get("ok") is True),
            "telemetry": lambda: {"ok": isinstance(telemetry.latest(), dict)},
        }
        self._lock = threading.Lock()
        self._cached_at = float("-inf")
        self._cached = None

    def collect(self, *, force=False) -> dict:
        now = self.clock()
        with self._lock:
            if not force and self._cached is not None and now - self._cached_at < self.cache_seconds:
                return json.loads(json.dumps(self._cached))
        started = self.clock()
        components = {}
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(self._probes))
        try:
            futures = {name: pool.submit(self._timed, probe) for name, probe in self._probes.items()}
            for name, future in futures.items():
                try:
                    components[name] = future.result(timeout=2.0)
                except Exception:
                    components[name] = {"ok": False, "elapsed_ms": 2000.0, "detail": "unavailable"}
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        features = self._features(components)
        required = ("conversation", "voice", "vision", "display", "matrix", "device_tools", "monitoring")
        ok = all(features[name] for name in required)
        core = all(components.get(name, {}).get("ok") for name in (
            "runtime", "media", "conversation_session", "local_llm",
        ))
        result = {
            "ok": ok,
            "status": "ready" if ok else "degraded" if core else "unavailable",
            "elapsed_ms": round((self.clock() - started) * 1000, 2),
            "components": components,
            "features": features,
        }
        with self._lock:
            self._cached_at = self.clock()
            self._cached = result
        return json.loads(json.dumps(result))

    def _timed(self, probe):
        started = self.clock()
        try:
            value = probe()
            if not isinstance(value, dict):
                value = {"ok": bool(value)}
            result = dict(value)
            result["ok"] = result.get("ok") is True
            result.setdefault("detail", "ready" if result["ok"] else "unavailable")
        except Exception:
            result = {"ok": False, "detail": "unavailable"}
        result["elapsed_ms"] = round((self.clock() - started) * 1000, 2)
        return result

    def _runtime(self):
        device_result = self.device.status(timeout_ms=1_200)
        speech_result = self.speech.status()
        roles = {
            name: value.get("status")
            for name, value in (device_result.get("roles") or {}).items()
            if isinstance(value, dict)
        }
        return {
            "ok": device_result.get("healthy") is True and speech_result.get("ready") is True,
            "roles": roles, "speech_ready": speech_result.get("ready") is True,
        }

    def _media(self):
        value = self.media.call("list", owner="service:readiness")
        return {"ok": isinstance(value, dict) and isinstance(value.get("count"), int)}

    def _conversation(self):
        value = self.conversation.stats()
        return {
            "ok": isinstance(value.get("active_sessions"), int)
                  and value.get("persistent") is False,
        }

    @staticmethod
    def _valid_models(value):
        models = value.get("data") if isinstance(value, dict) else None
        return isinstance(models, list) and any(
            isinstance(item, dict) and item.get("id") == "qwen3_4b_instruct_2507"
            for item in models
        )

    @staticmethod
    def _http_json(url, validator):
        try:
            with urllib.request.urlopen(url, timeout=0.8) as response:
                value = json.loads(response.read(64 * 1024))
            return {"ok": bool(validator(value))}
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            return {"ok": False}

    @staticmethod
    def _vision():
        try:
            with socket.create_connection(("127.0.0.1", 1337), timeout=0.8):
                return {"ok": True}
        except OSError:
            return {"ok": False}

    @staticmethod
    def _features(components):
        runtime = components.get("runtime") or {}
        roles = runtime.get("roles") or {}
        component_ok = lambda name: components.get(name, {}).get("ok") is True
        role_ok = lambda name: roles.get(name) == "resolved"
        core = (component_ok("runtime") and component_ok("media")
                and component_ok("conversation_session"))
        return {
            "conversation": core and component_ok("local_llm"),
            "voice": core and runtime.get("speech_ready") is True
                     and role_ok("audio.primary_input") and role_ok("audio.primary_output"),
            "vision": core and component_ok("vision_runner")
                      and (role_ok("camera.front") or role_ok("camera.environment")),
            "display": component_ok("display"),
            "matrix": core and role_ok("expression.matrix"),
            "device_tools": core and component_ok("local_agent"),
            "monitoring": component_ok("telemetry"),
        }
