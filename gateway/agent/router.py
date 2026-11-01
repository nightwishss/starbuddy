"""Policy-constrained local-first model routing and bounded NPU admission."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass

from governance import PrivacyContext, PrivacyPolicy


class RouteError(RuntimeError):
    pass


class ResourceBusy(RouteError):
    pass


@dataclass(frozen=True)
class RouteResult:
    answer: str
    backend: str
    escalated: bool
    reason: str


class NPUAdmission:
    """One shared bounded gate for LLM/Vision until coexistence is accepted."""

    def __init__(self, slots=1):
        if slots <= 0:
            raise ValueError("NPU slots must be positive")
        self._gate = threading.BoundedSemaphore(slots)

    @contextmanager
    def acquire(self, timeout=0.1):
        if not self._gate.acquire(timeout=timeout):
            raise ResourceBusy("NPU resource is busy")
        try:
            yield
        finally:
            self._gate.release()


class ControlPlaneAdmission:
    """Cross-process replacement for the legacy process-local NPU semaphore."""

    def __init__(self, client, *, owner="phone-gateway", amount=100):
        self.client, self.owner, self.amount = client, owner, amount

    @contextmanager
    def acquire(self, timeout=0.1):
        # The control plane is immediate admission rather than a hidden queue;
        # timeout is retained for compatibility with the application contract.
        try:
            lease = self.client.acquire(
                self.owner, {"npu_compute": self.amount}, "gateway.npu",
                priority=70, ttl_seconds=max(2, min(120, round(timeout + 90))),
            )
        except Exception as exc:
            raise ResourceBusy("NPU resource is busy") from exc
        try:
            yield
        finally:
            try:
                self.client.release(lease["lease_id"], self.owner)
            except Exception:
                pass


class LocalFirstRouter:
    def __init__(self, local, cloud=None, *, privacy_policy=None, admission=None):
        self.local = local
        self.cloud = cloud
        self.privacy_policy = privacy_policy or PrivacyPolicy()
        self.admission = admission or NPUAdmission()

    def route(self, message, privacy, *, complexity="normal", side_effects_executed=False):
        if complexity not in {"normal", "high"}:
            raise ValueError("invalid complexity")
        cloud_allowed = self.privacy_policy.cloud_text_allowed(privacy)
        if complexity == "high" and cloud_allowed and self.cloud is not None:
            return RouteResult(self.cloud(message), "cloud-agent", True, "high complexity")
        try:
            with self.admission.acquire():
                answer = self.local(message)
            if not isinstance(answer, str) or not answer.strip():
                raise RouteError("local model returned no answer")
            return RouteResult(answer.strip(), "local-agent", False, "local preferred")
        except (RouteError, OSError, TimeoutError) as exc:
            if side_effects_executed:
                raise RouteError("fallback denied after side effect") from exc
            if not cloud_allowed or self.cloud is None:
                raise RouteError("local route unavailable and cloud denied") from exc
            return RouteResult(self.cloud(message), "cloud-agent", True, "local unavailable")
