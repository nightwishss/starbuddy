"""Opt-in shared Qwen lane; deadlines never imply native cancellation."""

from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager

from control_plane import ControlPlaneClient, ControlPlaneError


LOG = logging.getLogger("vq.llm-admission")


class LLMAdmissionError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class LLMPermit:
    """Only a complete generation response or verified cancel can acknowledge."""

    def __init__(self, token=None):
        self.token = token
        self.terminal = False

    def acknowledge_terminal(self):
        self.terminal = True


@contextmanager
def local_llm_admission(*, owner, kind="foreground", timeout=30, client=None, environ=None):
    """Immediate admission with no inference queue and no automatic retry.

    Call acknowledge_terminal after reading the complete native HTTP response,
    before parsing its model content. A transport error or incomplete response
    keeps the shared lane quarantined, including beyond its ordinary lease TTL.
    """
    if not isinstance(kind, str) or kind not in {"foreground", "background"}:
        raise ValueError("invalid language admission kind")
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or not 0 < timeout <= 300):
        raise ValueError("timeout must be finite and between 0 and 300 seconds")
    environment = os.environ if environ is None else environ
    if environment.get("VQ_LLM_ADMISSION") != "1":
        if kind == "background":
            raise LLMAdmissionError("admission_disabled", "background language admission is disabled")
        yield LLMPermit()
        return
    control = client if client is not None else ControlPlaneClient(timeout=.5)
    try:
        lane = control.llm_enter(owner, kind=kind, ttl_seconds=min(300, max(1, timeout + 5)))
        permit = LLMPermit(lane["token"])
    except ControlPlaneError as exc:
        raise LLMAdmissionError(exc.code, "local language admission is unavailable") from exc
    except (OSError, ValueError, KeyError) as exc:
        raise LLMAdmissionError("llm_admission_unavailable", "local language admission is unavailable") from exc
    try:
        yield permit
    finally:
        try:
            control.llm_finish(permit.token, owner, terminal=permit.terminal)
        except (ControlPlaneError, OSError, ValueError, KeyError):
            # The remote lease becomes quarantined on expiry if finish cannot
            # be delivered. Never reopen the lane by guessing termination.
            LOG.warning("llm_finish_unavailable terminal=%s", permit.terminal)
