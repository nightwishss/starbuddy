"""Deterministic data-egress and model-route privacy policy."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


def starbuddy_offline_only() -> bool:
    """Operator-only gate, checked at use time, without I/O or background work.

    Unset/explicit 0 preserves the development profile. Any other value fails
    closed, including a malformed setting. This is not an OS network firewall.
    """
    return os.environ.get("VQ_STARBUDDY_OFFLINE_ONLY", "0") != "0"


class DataLabel(str, Enum):
    PUBLIC = "public"
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    BIOMETRIC = "biometric"
    CHILD_PRIVATE = "child_private"


class Destination(str, Enum):
    LOCAL = "local"
    CLOUD_TEXT = "cloud_text"
    CLOUD_MEDIA = "cloud_media"


@dataclass(frozen=True)
class PrivacyContext:
    labels: frozenset[DataLabel] = field(default_factory=lambda: frozenset({DataLabel.PERSONAL}))
    local_only: bool = False
    guardian_cloud_consent: bool = False


@dataclass(frozen=True)
class PrivacyDecision:
    allowed: bool
    destination: Destination
    reason: str


class PrivacyPolicy:
    NEVER_CLOUD = frozenset({DataLabel.SENSITIVE, DataLabel.CHILD_PRIVATE})

    def decide(self, context: PrivacyContext, destination: Destination) -> PrivacyDecision:
        if destination == Destination.LOCAL:
            return PrivacyDecision(True, destination, "local processing")
        if starbuddy_offline_only():
            return PrivacyDecision(False, destination, "StarBuddy offline-only profile")
        if context.local_only or context.labels & self.NEVER_CLOUD:
            return PrivacyDecision(False, destination, "local-only data")
        if destination == Destination.CLOUD_MEDIA and DataLabel.BIOMETRIC in context.labels:
            return PrivacyDecision(False, destination, "raw biometric media cannot leave device")
        if not context.guardian_cloud_consent:
            return PrivacyDecision(False, destination, "guardian cloud consent required")
        return PrivacyDecision(True, destination, "explicit cloud consent")

    def cloud_text_allowed(self, context: PrivacyContext) -> bool:
        return self.decide(context, Destination.CLOUD_TEXT).allowed
