"""Resolve stable logical device roles without leaking enumeration order upward."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path

from runtime.errors import HardwareError, ResourceNotFoundError, SchemaValidationError

PROFILE_VERSION = 1
ROLE_PREFIXES = ("camera.", "audio.", "display.", "expression.", "input.")
SELECTOR_KEYS = frozenset({"stable_id", "stable_id_contains", "name_contains", "transport", "connector"})
DEFAULT_MATRIX_SOCKET = Path(os.environ.get("VQ_MATRIX_SOCKET", "/run/user/1000/vq-matrix-relay.sock"))


def default_profile_path() -> Path:
    configured = os.environ.get("VQ_HARDWARE_PROFILE")
    if configured:
        return Path(configured).expanduser()
    user = Path.home() / ".config" / "vq-platform" / "hardware-profile.json"
    return user if user.is_file() else Path(__file__).with_name("default_profile.json")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HardwareError("无法读取硬件Profile", capability="device.profile",
                            details={"path": str(path), "reason": type(exc).__name__}) from exc
    return value


def _validate_profile(profile: dict) -> dict:
    if not isinstance(profile, dict) or profile.get("version") != PROFILE_VERSION:
        raise SchemaValidationError("硬件Profile版本无效", capability="device.profile")
    if not isinstance(profile.get("profile_id"), str) or not profile["profile_id"]:
        raise SchemaValidationError("硬件Profile缺少profile_id", capability="device.profile")
    roles = profile.get("roles")
    if not isinstance(roles, dict) or not roles:
        raise SchemaValidationError("硬件Profile必须包含roles", capability="device.profile")
    for role, spec in roles.items():
        if not isinstance(role, str) or not role.startswith(ROLE_PREFIXES):
            raise SchemaValidationError(f"逻辑设备角色无效: {role}", capability="device.profile")
        if not isinstance(spec, dict) or not isinstance(spec.get("kind"), str):
            raise SchemaValidationError(f"角色配置无效: {role}", capability="device.profile")
        selector = spec.get("selector")
        if not isinstance(selector, dict) or not selector:
            raise SchemaValidationError(f"角色缺少selector: {role}", capability="device.profile")
        unknown = set(selector) - SELECTOR_KEYS
        if unknown or any(not isinstance(value, str) or not value for value in selector.values()):
            raise SchemaValidationError(f"角色selector无效: {role}", capability="device.profile",
                                        details={"unknown": sorted(unknown)})
        if "settings" in spec and not isinstance(spec["settings"], dict):
            raise SchemaValidationError(f"角色settings必须是object: {role}", capability="device.profile")
    return profile


def _camera_inventory(root=Path("/dev/v4l/by-id")) -> list[dict]:
    result = []
    if not root.is_dir():
        return result
    for link in sorted(root.glob("*-video-index0")):
        try:
            device = link.resolve(strict=True)
        except OSError:
            continue
        stable_id = link.name
        result.append({
            "device_id": "camera:" + hashlib.sha256(stable_id.encode()).hexdigest()[:12],
            "kind": "camera", "stable_id": stable_id,
            "name": stable_id.removeprefix("usb-").removesuffix("-video-index0").replace("_", " "),
            "transport": "usb", "connected": True,
            "endpoint": str(device), "stable_path": str(link),
        })
    return result


def _pactl_inventory(command=None) -> list[dict]:
    command = command or subprocess.run
    result = []
    for object_type, kind in (("sources", "audio_input"), ("sinks", "audio_output")):
        try:
            completed = command(["pactl", "-f", "json", "list", object_type],
                                capture_output=True, text=True, timeout=5, check=False)
            values = json.loads(completed.stdout) if completed.returncode == 0 else []
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            values = []
        for item in values:
            props = item.get("properties") or {}
            if kind == "audio_input" and props.get("device.class") == "monitor":
                continue
            name = item.get("name") or props.get("node.name")
            if not name:
                continue
            transport = "usb" if "usb-" in name or str(props.get("alsa.components", "")).startswith("USB") else "platform"
            result.append({
                "device_id": f"{kind}:{name}", "kind": kind, "stable_id": name,
                "name": item.get("description") or props.get("device.description") or name,
                "transport": transport, "connected": True, "endpoint": name,
                "state": str(item.get("state", "unknown")).lower(),
                "sample_specification": item.get("sample_specification", ""),
                "muted": bool(item.get("mute", False)),
            })
    return result


def _display_inventory(root=Path("/sys/class/drm")) -> list[dict]:
    result = []
    for status in sorted(root.glob("card*-*/status")) if root.is_dir() else []:
        connector = status.parent.name.split("-", 1)[1]
        try:
            state = status.read_text().strip()
        except OSError:
            continue
        result.append({
            "device_id": f"display:{connector}", "kind": "display",
            "stable_id": connector, "connector": connector, "name": connector,
            "transport": "drm", "connected": state == "connected", "state": state,
            "endpoint": str(status.parent),
        })
    return result


def _matrix_inventory(path=DEFAULT_MATRIX_SOCKET) -> list[dict]:
    connected = False
    if path.exists():
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(0.25)
                client.connect(str(path))
                client.sendall(b'{"op":"status"}\n')
                response = json.loads(client.makefile("rb").readline(4097))
            connected = bool(response.get("ok") and response.get("result", {}).get("ready"))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return [{
        "device_id": "matrix:mcu-8x13", "kind": "matrix",
        "stable_id": "matrix:mcu-8x13", "name": "VENTUNO Q 8x13 MCU matrix",
        "transport": "mcu-relay", "connected": connected, "endpoint": str(path),
    }]


def discover_inventory() -> list[dict]:
    return _camera_inventory() + _pactl_inventory() + _display_inventory() + _matrix_inventory()


def _matches(device: dict, selector: dict) -> bool:
    for key, expected in selector.items():
        if key == "stable_id_contains":
            if expected not in str(device.get("stable_id", "")):
                return False
        elif key == "name_contains":
            if expected.casefold() not in str(device.get("name", "")).casefold():
                return False
        elif str(device.get(key, "")) != expected:
            return False
    return True


class HardwareProfileManager:
    def __init__(self, profile_path=None, inventory_provider=None):
        self.profile_path = Path(profile_path or default_profile_path())
        self.inventory_provider = inventory_provider or discover_inventory

    def profile(self) -> dict:
        profile = _validate_profile(_read_json(self.profile_path))
        return {**profile, "path": str(self.profile_path)}

    def inventory(self) -> list[dict]:
        return self.inventory_provider()

    def resolve_all(self) -> dict:
        profile = self.profile()
        inventory = self.inventory()
        roles = {}
        for role, spec in profile["roles"].items():
            matches = [item for item in inventory
                       if item.get("kind") == spec["kind"] and _matches(item, spec["selector"])]
            connected = [item for item in matches if item.get("connected")]
            status = "resolved" if len(connected) == 1 else "ambiguous" if len(connected) > 1 else "missing"
            roles[role] = {
                "role": role, "kind": spec["kind"], "required": bool(spec.get("required", False)),
                "status": status, "settings": dict(spec.get("settings") or {}),
                "device": connected[0] if len(connected) == 1 else None,
                "match_count": len(connected),
            }
        required_failures = sorted(role for role, value in roles.items()
                                   if value["required"] and value["status"] != "resolved")
        return {"profile_id": profile["profile_id"], "healthy": not required_failures,
                "required_failures": required_failures, "roles": roles}

    def resolve(self, role: str) -> dict:
        roles = self.resolve_all()["roles"]
        if role not in roles:
            raise ResourceNotFoundError(f"逻辑设备角色不存在: {role}", capability="device.resolve")
        value = roles[role]
        if value["status"] != "resolved":
            raise ResourceNotFoundError(f"逻辑设备角色未解析: {role}", capability="device.resolve",
                                        details={"status": value["status"], "match_count": value["match_count"]})
        return value
