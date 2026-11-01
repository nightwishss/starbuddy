"""Lifecycle owner for the optional PipeWire WebRTC echo-cancel graph."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from device_management import HardwareProfileManager


AEC_SOURCE_NAME = "vq_echo_cancelled_input"
AEC_SINK_NAME = "vq_echo_reference_output"


class EchoCancelError(RuntimeError):
    pass


class EchoCancelService:
    def __init__(self, *, manager=None, command=subprocess.run, state_path=None,
                 sleep=time.sleep):
        self.manager = manager or HardwareProfileManager()
        self.command = command
        self.sleep = sleep
        runtime = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
        self.state_path = Path(state_path or runtime / "vq-aec" / "state.json")
        self.module_id: str | None = None
        self.input_device_id: str | None = None
        self.output_device_id: str | None = None

    def _run(self, command: list[str], *, timeout=5, check=True):
        try:
            return self.command(
                command, capture_output=True, text=True, timeout=timeout, check=check,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise EchoCancelError(f"command failed: {command[0]}") from exc

    def _state(self, status: str, **extra) -> dict:
        value = {
            "status": status,
            "mode": "optional_aec_graph",
            "input_role": "audio.echo_cancelled_input",
            "output_role": "audio.echo_reference_output",
            "fallback_input_role": "audio.primary_input",
            "fallback_output_role": "audio.primary_output",
            "input_device_id": self.input_device_id,
            "output_device_id": self.output_device_id,
            **extra,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.parent.chmod(0o700)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.chmod(0o600)
        os.replace(temporary, self.state_path)
        return value

    def _virtual_nodes(self) -> tuple[bool, bool]:
        inventory = self.manager.inventory()
        source = any(
            item.get("kind") == "audio_input" and item.get("stable_id") == AEC_SOURCE_NAME
            for item in inventory
        )
        sink = any(
            item.get("kind") == "audio_output" and item.get("stable_id") == AEC_SINK_NAME
            for item in inventory
        )
        return source, sink

    def start(self) -> dict:
        if self.module_id is not None:
            raise EchoCancelError("AEC graph is already owned by this service")
        if any(self._virtual_nodes()):
            raise EchoCancelError("AEC virtual node already exists without service ownership")
        source = self.manager.resolve("audio.primary_input")["device"]
        sink = self.manager.resolve("audio.primary_output")["device"]
        self.input_device_id = source.get("device_id")
        self.output_device_id = sink.get("device_id")
        completed = self._run([
            "pactl", "load-module", "module-echo-cancel", "aec_method=webrtc",
            f"source_master={source['endpoint']}", f"sink_master={sink['endpoint']}",
            f"source_name={AEC_SOURCE_NAME}", f"sink_name={AEC_SINK_NAME}",
            "source_properties=device.description=VQ_Echo_Cancelled_Input",
            "sink_properties=device.description=VQ_Echo_Reference_Output",
        ])
        module_id = completed.stdout.strip()
        if not module_id.isdigit():
            raise EchoCancelError("echo-cancel module did not return an id")
        self.module_id = module_id
        for _ in range(20):
            if self._virtual_nodes() == (True, True):
                return self._state("ready", module_loaded=True, roles_resolved=True)
            self.sleep(0.1)
        self.stop()
        raise EchoCancelError("AEC virtual nodes did not become ready")

    def healthy(self) -> bool:
        if self.module_id is None or self._virtual_nodes() != (True, True):
            return False
        try:
            source = self.manager.resolve("audio.primary_input")["device"]
            sink = self.manager.resolve("audio.primary_output")["device"]
        except Exception:
            return False
        return (
            source.get("device_id") == self.input_device_id
            and sink.get("device_id") == self.output_device_id
        )

    def stop(self) -> dict:
        module_id, self.module_id = self.module_id, None
        if module_id:
            self._run(["pactl", "unload-module", module_id], check=False)
        return self._state("inactive", module_loaded=False, roles_resolved=False)
