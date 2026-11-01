"""Single-owner, latest-frame camera stream service."""

from .client import CameraStreamClient, CameraStreamError

__all__ = ["CameraStreamClient", "CameraStreamError"]
