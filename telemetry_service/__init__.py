"""Bounded in-memory platform telemetry and hardware-health service."""

from .client import TelemetryClient, TelemetryClientError
from .collector import TelemetryCollector

__all__ = ["TelemetryClient", "TelemetryClientError", "TelemetryCollector"]
