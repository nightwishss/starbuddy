"""One admitted cold-model preparation, separate from user interaction latency.

There is no inference retry, persistent state, quarantine recovery, native
process launch, media access, or framework bootstrap here. Probes are bounded
public status reads; only the client owns normal admission and terminal ACKs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import time


NATIVE_TIMEOUT_SECONDS = 65.0
HEALTH_WAIT_SECONDS = 10.0
STARTUP_HEALTH_WAIT_SECONDS = 45.0
STARTUP_QUIET_SECONDS = 11.0
WARMUP_TEXT = "请只回复一个字：好。"
MAX_STATUS_BYTES = 16384


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _idle_reason(value, startup):
    if type(value) is not dict:
        return "invalid_status"
    lane, output, interaction = (value.get(key) for key in ("lane", "output", "interaction"))
    if (type(lane) is not dict or not {"state", "lane", "persistence_error"} <= lane.keys()
            or type(output) is not dict
            or not {"ready", "current", "queued", "output_quarantined"} <= output.keys()
            or type(output.get("queued")) is not int
            or type(output.get("ready")) is not bool
            or type(output.get("output_quarantined")) is not bool):
        return "invalid_status"
    if lane["state"] == "quarantined" or lane["persistence_error"] is not None:
        return "lane_quarantined"
    if lane["state"] != "idle" or lane["lane"] is not None:
        return "lane_busy"
    if output["output_quarantined"] or output.get("tts_quarantined", False) is not False:
        return "output_quarantined"
    if not output["ready"] or output["current"] is not None or output["queued"] != 0:
        return "output_busy"
    if type(interaction) is not dict:
        return "invalid_status"
    if interaction.get("availability") == "not_started":
        if interaction.get("reason") not in {"connection_refused", "not_found"}:
            return "invalid_status"
        return None if startup else "interaction_unavailable"
    state = interaction.get("state")
    if interaction.get("availability") != "available" or type(state) is not dict:
        return "invalid_status"
    if not {"phase", "user_busy", "output_pending", "quiet"} <= state.keys():
        return "invalid_status"
    if (state["phase"] != "idle" or state["user_busy"] is not False
            or state["output_pending"] is not False or state["quiet"] is not False):
        return "interaction_busy"
    return None


def prepare_once(*, client_factory, health_probe, idle_probe, startup=False,
                 clock=time.monotonic, sleep=time.sleep, health_wait_seconds=None):
    """Prepare at most once; injected probes receive a remaining-time budget.

    ``health_probe(timeout)`` returns exactly True on an available native HTTP
    API. ``idle_probe(timeout)`` returns lane/output/interaction status. Startup
    has a separate bounded cold-start API budget and requires an 11-second
    fixed quiet window. An injected health budget can shorten either mode's
    wait, but cannot extend it. Model-call counters count client attempts, not
    unobservable native executions.
    """
    if type(startup) is not bool:
        raise ValueError("invalid preparation bounds")
    maximum_health_wait = STARTUP_HEALTH_WAIT_SECONDS if startup else HEALTH_WAIT_SECONDS
    if health_wait_seconds is None:
        health_wait_seconds = maximum_health_wait
    if (not _finite(health_wait_seconds)
            or not 0 < health_wait_seconds <= maximum_health_wait):
        raise ValueError("invalid preparation bounds")
    started = clock()
    timing = {}
    attempts = 0
    checks = 0
    stage = "api_health_wait"

    def result(status, reason):
        return {"status": status, "ready": status == "ready", "stage": stage,
                "reason": reason, "startup": startup, "admission_kind": "background",
                "model_calls": attempts, "counter_scope": "client_attempts_not_native_executions",
                "health_checks": checks, "native_budget_seconds": NATIVE_TIMEOUT_SECONDS,
                "health_budget_seconds": health_wait_seconds,
                "user_response_latency_measured": False,
                "timing_ms": {**timing, "total": round(max(0, clock() - started) * 1000, 2)}}

    def preflight(timeout):
        try:
            reason = _idle_reason(idle_probe(timeout), startup)
        except Exception:
            return "failure", "status_unavailable"
        if reason:
            return ("failure" if reason == "invalid_status" else "busy"), reason
        return None

    health_started = clock()
    health_deadline = health_started + health_wait_seconds
    healthy = False
    # The iteration bound also fails closed for a broken injected clock/sleep.
    for _ in range(math.ceil(health_wait_seconds / .25) + 1):
        remaining = health_deadline - clock()
        if remaining <= 0:
            break
        checks += 1
        try:
            healthy = health_probe(min(.5, remaining)) is True
        except Exception:
            healthy = False
        if clock() >= health_deadline:
            break
        if healthy:
            break
        sleep(min(.25, max(0, health_deadline - clock())))
    else:
        healthy = False
    timing[stage] = round(max(0, clock() - health_started) * 1000, 2)
    if not healthy or clock() >= health_deadline:
        return result("failure", "api_health_unavailable")

    if startup:
        stage = "startup_quiet_window"
        quiet_started = clock()
        quiet_deadline = quiet_started + STARTUP_QUIET_SECONDS
        for _ in range(23):
            remaining = quiet_deadline - clock()
            if remaining <= 0:
                break
            failed = preflight(min(1.5, remaining))
            if failed:
                timing[stage] = round(max(0, clock() - quiet_started) * 1000, 2)
                return result(*failed)
            sleep(min(.5, max(0, quiet_deadline - clock())))
        timing[stage] = round(max(0, clock() - quiet_started) * 1000, 2)
        if clock() < quiet_deadline:
            return result("failure", "quiet_window_incomplete")

    stage = "preflight"
    preflight_started = clock()
    failed = preflight(1.5)
    timing[stage] = round(max(0, clock() - preflight_started) * 1000, 2)
    if failed:
        return result(*failed)

    stage = "native_warmup"
    native_started = clock()
    try:
        client = client_factory(timeout=NATIVE_TIMEOUT_SECONDS, admission_kind="background")
        attempts = 1
        reply = client.chat(WARMUP_TEXT, max_completion_tokens=8, temperature=0.0)
    except Exception as exc:
        timing[stage] = round(max(0, clock() - native_started) * 1000, 2)
        # Admission refusal is expected contention, not evidence the model broke.
        cause = exc
        for _ in range(3):
            if getattr(cause, "code", None) in {"llm_busy", "llm_foreground_recent", "resource_busy"}:
                return result("busy", "background_admission_busy")
            cause = getattr(cause, "__cause__", None)
            if cause is None:
                break
        return result("failure", "native_warmup_failed")
    timing[stage] = round(max(0, clock() - native_started) * 1000, 2)
    if clock() - native_started >= NATIVE_TIMEOUT_SECONDS:
        return result("failure", "native_deadline_exceeded")
    if (type(reply) is not str or len(reply) > 64
            or reply.strip().rstrip("。.!！?？") != "好"):
        return result("failure", "native_response_invalid")
    # The completed body is deliberately neither returned nor persisted.
    stage = "postflight"
    postflight_started = clock()
    failed = preflight(1.5)
    timing[stage] = round(max(0, clock() - postflight_started) * 1000, 2)
    if failed:
        return result(*failed)
    stage = "complete"
    return result("ready", "native_request_completed")


def _remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise TimeoutError("status deadline")
    return value


def _http_get(port, path, timeout, *, headers_only=False):
    """Fixed public loopback routes with a total socket-read deadline.

    Native health only needs a 200 response header, not model readiness. The
    kiosk's bounded JSON status uses its existing Content-Length framing.
    """
    if (port, path) not in {(9001, "/v1/models"), (8765, "/api/interaction/state")}:
        raise ValueError("unapproved status endpoint")
    deadline = time.monotonic() + timeout
    with socket.create_connection(("127.0.0.1", port), timeout=_remaining(deadline)) as connection:
        connection.settimeout(_remaining(deadline))
        connection.sendall((f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n").encode("ascii"))
        raw = bytearray()
        while b"\r\n\r\n" not in raw:
            connection.settimeout(_remaining(deadline))
            part = connection.recv(min(4096, MAX_STATUS_BYTES + 1 - len(raw)))
            _remaining(deadline)
            if not part:
                raise ValueError("incomplete status header")
            raw.extend(part)
            if len(raw) > MAX_STATUS_BYTES:
                raise ValueError("status too large")
        header, body = bytes(raw).split(b"\r\n\r\n", 1)
        lines = header.split(b"\r\n")
        status_line = lines[0].split(b" ", 2)
        if len(status_line) < 2 or status_line[0] not in {b"HTTP/1.0", b"HTTP/1.1"}:
            raise ValueError("invalid status line")
        status = int(status_line[1])
        if headers_only or status != 200:
            return status, b""
        fields = {}
        for line in lines[1:]:
            key, separator, value = line.partition(b":")
            key = key.lower()
            if not separator or key in fields:
                raise ValueError("invalid status header")
            fields[key] = value.strip()
        if b"transfer-encoding" in fields:
            raise ValueError("unsupported status framing")
        length = int(fields[b"content-length"])
        if not 0 <= length <= MAX_STATUS_BYTES:
            raise ValueError("status too large")
        while len(body) < length:
            connection.settimeout(_remaining(deadline))
            part = connection.recv(min(4096, length - len(body)))
            _remaining(deadline)
            if not part:
                raise ValueError("incomplete status body")
            body += part
        if len(body) != length:
            raise ValueError("invalid status length")
        return status, body


def _unix_status(path, operation, timeout):
    if operation not in {"llm.status", "system.status"}:
        raise ValueError("unapproved status operation")
    name = "vq-control.sock" if operation == "llm.status" else "vq-output.sock"
    if path != f"/run/user/{os.getuid()}/{name}":
        raise ValueError("unapproved status socket")
    deadline = time.monotonic() + timeout
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(_remaining(deadline))
        connection.connect(path)
        connection.settimeout(_remaining(deadline))
        connection.sendall(json.dumps({"operation": operation}).encode("ascii") + b"\n")
        raw = bytearray()
        while b"\n" not in raw:
            connection.settimeout(_remaining(deadline))
            part = connection.recv(min(4096, MAX_STATUS_BYTES + 1 - len(raw)))
            _remaining(deadline)
            if not part:
                raise ValueError("incomplete status")
            raw.extend(part)
            if len(raw) > MAX_STATUS_BYTES:
                raise ValueError("status too large")
        if not raw.endswith(b"\n"):
            raise ValueError("unexpected status frames")
        value = json.loads(raw)
        if type(value) is not dict or value.get("ok") is not True or type(value.get("result")) is not dict:
            raise ValueError("status unavailable")
        return value["result"]


def _health_probe(timeout):
    return _http_get(9001, "/v1/models", timeout, headers_only=True)[0] == 200


def _interaction_probe(timeout):
    try:
        status, body = _http_get(8765, "/api/interaction/state", timeout)
    except ConnectionRefusedError:
        return {"availability": "not_started", "reason": "connection_refused"}
    if status == 404:
        return {"availability": "not_started", "reason": "not_found"}
    if status != 200:
        raise ValueError("interaction status unavailable")
    value = json.loads(body)
    if type(value) is not dict:
        raise ValueError("invalid interaction status")
    return {"availability": "available", "state": value}


def _idle_probe(timeout):
    deadline = time.monotonic() + timeout
    prefix = f"/run/user/{os.getuid()}"
    lane = _unix_status(prefix + "/vq-control.sock", "llm.status", min(.5, _remaining(deadline)))
    output = _unix_status(prefix + "/vq-output.sock", "system.status", min(.5, _remaining(deadline)))
    interaction = _interaction_probe(min(.5, _remaining(deadline)))
    return {"lane": lane, "output": output, "interaction": interaction}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", required=True)
    parser.add_argument("--startup", action="store_true")
    args = parser.parse_args(argv)
    # Import the existing client only for this explicit CLI invocation. Its
    # background admission fails closed if the normal shared lane is disabled.
    from gateway.llm.client import LocalLLMClient
    value = prepare_once(client_factory=LocalLLMClient, health_probe=_health_probe,
                         idle_probe=_idle_probe, startup=args.startup)
    # Foreground output/activity may legitimately occupy an otherwise healthy
    # API during service startup. Defer optional warmup rather than stop that
    # API; keep the busy/not-ready result and never retry or bypass admission.
    deferred = (args.startup and value.get("status") == "busy"
                and value.get("ready") is False
                and value.get("stage") in {"startup_quiet_window", "preflight"}
                and value.get("reason") in {"output_busy", "interaction_busy"}
                and type(value.get("model_calls")) is int and value["model_calls"] == 0)
    if deferred:
        value = {**value, "preparation_deferred": True}
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if deferred else {"ready": 0, "busy": 2, "failure": 1}[value["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
