"""Pure policy and Linux process probes for the bounded kiosk watchdog."""

from __future__ import annotations

import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse


EXPECTED_TITLE_PREFIX = "StarBuddy"


def notify_systemd(message: str) -> bool:
    """Notify from the GTK loop itself; a blocked loop must stop heartbeats."""
    # systemd-inhibit removes NOTIFY_SOCKET before starting its child. The unit
    # forwards that exact manager-provided value under a private alias.
    address = os.environ.get("NOTIFY_SOCKET") or os.environ.get("VQ_KIOSK_NOTIFY_SOCKET", "")
    if not address or address[0] not in {"/", "@"}:
        return False
    if address[0] == "@":
        address = "\0" + address[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
            channel.settimeout(0.05)
            channel.sendto(message.encode("ascii"), address)
        return True
    except OSError:
        return False


def title_is_expected(title: str) -> bool:
    return isinstance(title, str) and title.startswith(EXPECTED_TITLE_PREFIX)


@dataclass(frozen=True)
class RecoveryDecision:
    action: str
    reason: str
    delay_seconds: float = 0.0


class KioskRecoveryPolicy:
    """Bound reload attempts and convert persistent faults into process exit."""

    def __init__(self, *, max_reloads=2, min_reload_interval=10.0,
                 heartbeat_failures=3, backend_failures=3,
                 memory_soft_mb=1200.0, memory_breaches=6,
                 render_required=False, render_failures=3,
                 healthy_reset_seconds=60.0):
        if max_reloads < 1 or min_reload_interval < 1:
            raise ValueError("invalid reload policy")
        if min(heartbeat_failures, backend_failures, memory_breaches, render_failures) < 2:
            raise ValueError("failure thresholds must be at least two")
        if memory_soft_mb < 128:
            raise ValueError("memory threshold is too low")
        if healthy_reset_seconds < 1:
            raise ValueError("healthy reset interval must be at least one second")
        self.max_reloads = max_reloads
        self.min_reload_interval = min_reload_interval
        self.heartbeat_failure_threshold = heartbeat_failures
        self.backend_failure_threshold = backend_failures
        self.memory_soft_mb = memory_soft_mb
        self.memory_breach_threshold = memory_breaches
        self.failure_count = 0
        self.heartbeat_failure_count = 0
        self.backend_failure_count = 0
        self.memory_breach_count = 0
        self.reload_count = 0
        self.recovery_count = 0
        self.last_reload_at: float | None = None
        self.page_healthy = False
        self.backend_healthy = False
        self.render_required = render_required
        self.render_healthy = False
        self.render_failure_count = 0
        self.render_failure_threshold = render_failures
        self.healthy_reset_seconds = healthy_reset_seconds
        self._healthy_since: float | None = None

    def navigation(self) -> None:
        # A successful probe on an old page must not forgive a failed recovery.
        self.page_healthy = False
        self._healthy_since = None
        self.render_unavailable()

    def render_unavailable(self) -> None:
        # Hidden/unmapped pages cannot prove sustained rendering health.
        self.render_healthy = False
        self.render_failure_count = 0
        if self.render_required:
            self._healthy_since = None

    def render(self, success: bool, now: float) -> RecoveryDecision | None:
        self.render_healthy = success
        if success:
            self.render_failure_count = 0
            self._reset_recovery_budget_if_healthy(now)
            return None
        if self.render_required:
            self._healthy_since = None
        self.render_failure_count += 1
        if not self.render_required or self.render_failure_count < self.render_failure_threshold:
            return None
        self.render_failure_count = 0
        return self.failure("render_stall", now)

    def failure(self, reason: str, now: float) -> RecoveryDecision:
        self.page_healthy = False
        self._healthy_since = None
        self.failure_count += 1
        self.recovery_count += 1
        if self.reload_count >= self.max_reloads:
            return RecoveryDecision("exit", reason)
        earliest = now + self.min_reload_interval
        if self.last_reload_at is not None:
            earliest = max(earliest, self.last_reload_at + self.min_reload_interval)
        self.reload_count += 1
        self.last_reload_at = earliest
        return RecoveryDecision("reload", reason, earliest - now)

    def heartbeat(self, success: bool, now: float) -> RecoveryDecision | None:
        if success:
            self.page_healthy = True
            self.heartbeat_failure_count = 0
            self._reset_recovery_budget_if_healthy(now)
            return None
        self.page_healthy = False
        self._healthy_since = None
        self.heartbeat_failure_count += 1
        if self.heartbeat_failure_count < self.heartbeat_failure_threshold:
            return None
        return self.failure("heartbeat", now)

    def backend(self, success: bool, now: float) -> RecoveryDecision | None:
        if success:
            self.backend_healthy = True
            self.backend_failure_count = 0
            self._reset_recovery_budget_if_healthy(now)
            return None
        self.backend_healthy = False
        self._healthy_since = None
        self.backend_failure_count += 1
        if self.backend_failure_count < self.backend_failure_threshold:
            return None
        return self.failure("backend_health", now)

    def memory(self, total_rss_mb: float, now: float) -> RecoveryDecision | None:
        if total_rss_mb <= self.memory_soft_mb:
            self.memory_breach_count = 0
            return None
        self.memory_breach_count += 1
        if self.memory_breach_count < self.memory_breach_threshold:
            return None
        self.memory_breach_count = 0
        self.recovery_count += 1
        return RecoveryDecision("exit", "memory_soft_limit")

    def mark_healthy(self, now: float | None = None) -> None:
        self.page_healthy = True
        self.backend_healthy = True
        self.heartbeat_failure_count = 0
        self.backend_failure_count = 0
        self._reset_recovery_budget_if_healthy(time.monotonic() if now is None else now)

    def _reset_recovery_budget_if_healthy(self, now: float) -> None:
        if not self.page_healthy or not self.backend_healthy:
            self._healthy_since = None
            return
        if self.render_required and not self.render_healthy:
            self._healthy_since = None
            return
        # A briefly responsive page must not grant an endless series of reloads.
        # Use probe timestamps: no extra timer, thread, or polling is needed.
        if self._healthy_since is None:
            self._healthy_since = now
        if now - self._healthy_since < self.healthy_reset_seconds:
            return
        self.failure_count = 0
        self.reload_count = 0
        self.last_reload_at = None

    def status(self) -> dict:
        return {
            "page_healthy": self.page_healthy,
            "backend_healthy": self.backend_healthy,
            "render_healthy": self.render_healthy,
            "render_required": self.render_required,
            "render_failure_count": self.render_failure_count,
            "failure_count": self.failure_count,
            "heartbeat_failure_count": self.heartbeat_failure_count,
            "backend_failure_count": self.backend_failure_count,
            "memory_breach_count": self.memory_breach_count,
            "reload_count": self.reload_count,
            "recovery_count": self.recovery_count,
        }


def health_url_for(page_url: str) -> str:
    """Map a supported local product page to its own readiness endpoint."""
    parsed = urlparse(page_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("kiosk endpoints must use loopback HTTP")
    port = parsed.port or 80
    health_path = {8795: "/status", 8765: "/healthz"}.get(port, "/")
    return urlunparse(parsed._replace(path=health_path, params="", query="", fragment=""))


def probe_url(url: str, timeout=2.0) -> bool:
    request = urllib.request.Request(url, headers={"Connection": "close"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read(256)
            return response.status == 200
    except (OSError, TimeoutError, urllib.error.URLError):
        return False


def wait_for_urls(urls, timeout, *, interval=1.0, clock=time.monotonic,
                  sleep=time.sleep, probe=probe_url) -> bool:
    deadline = clock() + timeout
    while True:
        if all(probe(url) for url in urls):
            return True
        remaining = deadline - clock()
        if remaining <= 0:
            return False
        sleep(min(interval, remaining))


def process_memory_snapshot(pid=None, proc_root=Path("/proc")) -> dict:
    """Return RSS for the kiosk process and its Linux child process tree."""
    root = Path(proc_root)
    root_pid = os.getpid() if pid is None else int(pid)
    pending = [root_pid]
    seen = set()
    rows = []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        process = root / str(current)
        try:
            name = (process / "comm").read_text(encoding="utf-8").strip()
            status = (process / "status").read_text(encoding="utf-8")
        except OSError:
            continue
        rss_kb = 0
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                try:
                    rss_kb = int(line.split()[1])
                except (ValueError, IndexError):
                    pass
                break
        rows.append((current, name, rss_kb))
        try:
            children = (process / "task" / str(current) / "children").read_text(
                encoding="utf-8").split()
            pending.extend(int(value) for value in children)
        except (OSError, ValueError):
            pass
    main_kb = next((rss for process_id, _name, rss in rows if process_id == root_pid), 0)
    web_rows = [row for row in rows if row[1].startswith("WebKitWeb")]
    return {
        "main_rss_mb": round(main_kb / 1024, 2),
        "web_process_rss_mb": round(sum(row[2] for row in web_rows) / 1024, 2),
        "web_process_count": len(web_rows),
        "total_rss_mb": round(sum(row[2] for row in rows) / 1024, 2),
        "process_count": len(rows),
    }
