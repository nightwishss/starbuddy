"""Cooperative Pose degradation against the existing Control Plane budget.

No frame queue, model invocation or lease eviction. Already-running native work
is never declared cancelled. The shared QNN worker remains the serialization
boundary between Pose and TTS; this gate reduces new low-priority submissions.
"""
import math
import os
import time


class PoseAdmission:
    def __init__(self, client=None, *, enabled=None, clock=time.monotonic):
        self.enabled = (os.environ.get("VQ_POSE_RESOURCE_ADMISSION", "0") == "1"
                        if enabled is None else enabled)
        self.client, self.clock = client, clock
        self.next_probe = 0.0
        self.allowed = False
        self.reason = "not_sampled" if self.enabled else "disabled"
        self.probes = self.deferred = self.failures = 0

    def admit(self):
        if not self.enabled:
            return True
        now = self.clock()
        if now >= self.next_probe:
            self.probes += 1
            try:
                if self.client is None:
                    from control_plane import ControlPlaneClient
                    self.client = ControlPlaneClient(timeout=.05)
                value = self.client.status()
                available = value["available"]["npu_compute"]
                if (isinstance(available, bool) or not isinstance(available, (int, float))
                        or not math.isfinite(available) or not 0 <= available <= 100):
                    raise ValueError("invalid resource budget")
                # 35 is the existing registry Pose estimate, not utilization.
                self.allowed = available >= 35
                self.reason = "available" if self.allowed else "foreground_compute"
                self.next_probe = now + .25
            except Exception:
                self.allowed, self.reason = False, "control_unavailable"
                self.failures += 1
                self.next_probe = now + 1.0
        if not self.allowed:
            self.deferred += 1
        return self.allowed

    def status(self):
        return {"enabled": self.enabled, "reason": self.reason,
                "probes": self.probes, "deferred_frames": self.deferred,
                "probe_failures": self.failures, "policy": "cooperative-skip-no-queue",
                "utilization_measured": False}
