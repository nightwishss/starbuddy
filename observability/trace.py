"""In-memory timing traces that never store prompts, transcripts or media."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from contextlib import contextmanager


DEFAULT_BUDGETS = {
    "physical-voice": {
        "total_ms": 30000,
        "stages": {"capture": 24000, "stt": 6000, "decision": 8000,
                   "tts": 8000, "physical-playback-start": 1500},
    },
    "voice": {
        "total_ms": 22000,
        "stages": {"stt": 6000, "decision": 8000, "tts": 8000},
    },
    "chat": {
        "total_ms": 16000,
        "stages": {"decision": 8000, "tts": 8000},
    },
}


class TraceStore:
    def __init__(self, *, max_traces=256, ttl_seconds=3600, clock=time.monotonic,
                 budgets=None):
        self.max_traces = max_traces
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self.budgets = budgets or DEFAULT_BUDGETS
        self._traces = OrderedDict()
        self._lock = threading.Lock()

    def start(self, trace_id: str, kind: str) -> None:
        now = self.clock()
        with self._lock:
            self._purge(now)
            while len(self._traces) >= self.max_traces:
                self._traces.popitem(last=False)
            self._traces[trace_id] = {"kind": kind, "started": now, "updated": now,
                                      "stages": [], "ok": None, "backend": None}

    def set_backend(self, trace_id: str, backend: str) -> None:
        allowed = {"local-qwen", "local-agent", "safety-policy", "cloud-agent", "hermes-agent"}
        if backend not in allowed:
            backend = "other"
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is not None:
                trace["backend"] = backend

    @contextmanager
    def stage(self, trace_id: str, name: str):
        started = self.clock()
        ok = False
        try:
            yield
            ok = True
        finally:
            ended = self.clock()
            with self._lock:
                trace = self._traces.get(trace_id)
                if trace is not None:
                    trace["updated"] = ended
                    trace["stages"].append({"name": name, "elapsed_ms": round((ended - started) * 1000, 2), "ok": ok})

    def finish(self, trace_id: str, ok: bool) -> dict:
        now = self.clock()
        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                raise KeyError(trace_id)
            trace["updated"] = now
            trace["ok"] = bool(ok)
            return self._public(trace_id, trace)

    def get(self, trace_id: str) -> dict | None:
        with self._lock:
            self._purge(self.clock())
            trace = self._traces.get(trace_id)
            return self._public(trace_id, trace) if trace is not None else None

    def summary(self) -> dict:
        with self._lock:
            self._purge(self.clock())
            finished = [self._public(trace_id, trace) for trace_id, trace in self._traces.items()
                        if trace["ok"] is not None]
        kinds = {}
        for kind in sorted({item["kind"] for item in finished}):
            traces = [item for item in finished if item["kind"] == kind]
            elapsed = [item["elapsed_ms"] for item in traces]
            stage_values = {}
            for trace in traces:
                for stage in trace["stages"]:
                    stage_values.setdefault(stage["name"], []).append(stage["elapsed_ms"])
            kinds[kind] = {
                "count": len(traces),
                "success_rate": round(sum(item["ok"] for item in traces) / len(traces), 4),
                "budget_met_rate": round(
                    sum(item["budget"]["met"] for item in traces) / len(traces), 4),
                "elapsed_ms": self._distribution(elapsed),
                "stages": {name: self._distribution(values)
                           for name, values in sorted(stage_values.items())},
                "backends": {
                    backend: len([item for item in traces if (item.get("backend") or "unknown") == backend])
                    for backend in sorted({item.get("backend") or "unknown" for item in traces})
                },
            }
        return {"finished": len(finished), "capacity": self.max_traces, "kinds": kinds}

    def _public(self, trace_id, trace):
        elapsed_ms = round((trace["updated"] - trace["started"]) * 1000, 2)
        stages = [dict(item) for item in trace["stages"]]
        budget = self._budget(trace["kind"], elapsed_ms, stages, trace["ok"])
        return {"trace_id": trace_id, "kind": trace["kind"], "backend": trace.get("backend"),
                "elapsed_ms": elapsed_ms,
                "ok": trace["ok"], "stages": stages, "budget": budget}

    def _budget(self, kind, elapsed_ms, stages, finished):
        policy = self.budgets.get(kind) or {}
        violations = []
        total_limit = policy.get("total_ms")
        if total_limit is not None and elapsed_ms > total_limit:
            violations.append({"name": "total", "elapsed_ms": elapsed_ms, "limit_ms": total_limit})
        stage_limits = policy.get("stages") or {}
        for stage in stages:
            limit = stage_limits.get(stage["name"])
            if limit is not None and stage["elapsed_ms"] > limit:
                violations.append({"name": stage["name"], "elapsed_ms": stage["elapsed_ms"],
                                   "limit_ms": limit})
        return {"met": None if finished is None else not violations,
                "total_limit_ms": total_limit, "violations": violations}

    @staticmethod
    def _distribution(values):
        ordered = sorted(values)
        if not ordered:
            return {"p50": 0.0, "p95": 0.0, "max": 0.0}
        def percentile(fraction):
            index = max(0, min(len(ordered) - 1, int((len(ordered) * fraction) + 0.999999) - 1))
            return ordered[index]
        return {"p50": percentile(0.50), "p95": percentile(0.95), "max": ordered[-1]}

    def _purge(self, now):
        for key in [key for key, value in self._traces.items() if now - value["updated"] >= self.ttl_seconds]:
            self._traces.pop(key, None)
