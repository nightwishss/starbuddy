"""Rate-limited fault records in the existing logger/journal, not a metrics daemon."""

import threading
import time
from collections import OrderedDict


class FaultLog:
    """Bounded per-component counters; never logs exception text or payloads.

    Event/action are static names supplied by code, not user content. No timers,
    network calls, background workers, or persistent user data are created.
    """

    def __init__(self, logger, *, interval=60.0, capacity=32, clock=time.monotonic):
        if interval < 1 or not 1 <= capacity <= 256:
            raise ValueError("invalid fault log bounds")
        self.logger, self.interval, self.capacity, self.clock = logger, interval, capacity, clock
        self._entries = OrderedDict()
        self._lock = threading.Lock()

    def record(self, event, *, action, error=None):
        now = self.clock()
        with self._lock:
            if event not in self._entries:
                if len(self._entries) >= self.capacity:
                    self._entries.popitem(last=False)
                self._entries[event] = {"count": 0, "suppressed": 0, "last_log": float("-inf")}
            entry = self._entries[event]
            entry["count"] += 1
            if now - entry["last_log"] < self.interval:
                entry["suppressed"] += 1
                return
            count, suppressed = entry["count"], entry["suppressed"]
            entry["last_log"], entry["suppressed"] = now, 0
        self.logger.warning("fault event=%s action=%s error=%s count=%d suppressed=%d",
                            event, action, type(error).__name__ if error is not None else "none",
                            count, suppressed)
