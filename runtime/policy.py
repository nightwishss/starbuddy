"""本地Runtime的显式权限策略。"""

import os

from runtime.errors import PermissionDeniedError


DEFAULT_PERMISSIONS = frozenset({
    "service.system.read",
    "platform.device.read",
    "service.file.read",
    "service.file.write",
    "device.gpio.read",
    "device.led.read",
    "device.led.write",
    "device.matrix.read",
    "device.matrix.write",
    "device.display.read",
    "device.display.write",
    "device.audio.read",
    "device.audio.write",
    "device.audio.capture",
    "device.audio.playback",
    "device.camera.read",
    "device.camera.capture",
    "ai.speech.execute",
    "ai.vision.execute",
})


class AccessPolicy:
    def __init__(self, owner_uid=None, allowed_permissions=None):
        self.owner_uid = os.getuid() if owner_uid is None else owner_uid
        self.allowed_permissions = frozenset(
            DEFAULT_PERMISSIONS if allowed_permissions is None else allowed_permissions)

    def authorize_peer(self, peer_uid):
        if peer_uid != self.owner_uid:
            raise PermissionDeniedError(
                f"调用方uid={peer_uid}无权访问uid={self.owner_uid}的Runtime",
                details={"peer_uid": peer_uid, "owner_uid": self.owner_uid})

    def authorize_capability(self, capability, metadata):
        required = set(metadata.get("permissions") or [])
        missing = sorted(required - self.allowed_permissions)
        if missing:
            raise PermissionDeniedError(
                f"Capability缺少权限: {', '.join(missing)}",
                capability=capability, details={"missing": missing})
