"""Trusted application adapter for explicit, data-minimized feedback."""

from .client import FeedbackClient, FeedbackClientError

__all__ = ["FeedbackClient", "FeedbackClientError"]
