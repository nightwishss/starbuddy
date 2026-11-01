"""Run the optional echo-cancel graph until stopped or its hardware disappears."""

from __future__ import annotations

import logging
import signal
import threading

from .echo_cancel import EchoCancelError, EchoCancelService


LOG = logging.getLogger("vq.aec")


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    stopped = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _signum, _frame: stopped.set())
    service = EchoCancelService()
    try:
        service.start()
        failures = 0
        while not stopped.wait(2.0):
            if service.healthy():
                failures = 0
                continue
            failures += 1
            if failures >= 3:
                raise EchoCancelError("AEC graph or physical roles became unavailable")
        return 0
    except EchoCancelError as exc:
        LOG.error("AEC service unavailable: %s", exc)
        return 1
    finally:
        service.stop()


if __name__ == "__main__":
    raise SystemExit(main())
