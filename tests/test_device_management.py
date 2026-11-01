#!/usr/bin/env python3
"""Hardware profile validation, stable resolution and failure tests."""

import json
import tempfile
import unittest
from pathlib import Path

from device_management.manager import HardwareProfileManager, _validate_profile
from runtime.errors import ResourceNotFoundError, SchemaValidationError


class HardwareProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "profile.json"
        self.profile = {
            "version": 1, "profile_id": "test-board",
            "roles": {
                "camera.front": {"kind": "camera", "required": True,
                                 "selector": {"stable_id_contains": "SERIAL-A"}},
                "audio.primary_output": {"kind": "audio_output", "required": False,
                                         "selector": {"name_contains": "USB Speaker", "transport": "usb"},
                                         "settings": {"volume_percent": 80}},
            },
        }
        self.path.write_text(json.dumps(self.profile), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def manager(self, devices):
        return HardwareProfileManager(self.path, inventory_provider=lambda: devices)

    def test_resolves_stable_role_and_settings(self):
        devices = [
            {"device_id": "camera:1", "kind": "camera", "stable_id": "CAM_SERIAL-A", "name": "Front", "transport": "usb", "connected": True},
            {"device_id": "audio:1", "kind": "audio_output", "stable_id": "node-1", "name": "USB Speaker", "transport": "usb", "connected": True},
        ]
        result = self.manager(devices).resolve_all()
        self.assertTrue(result["healthy"])
        self.assertEqual(result["roles"]["camera.front"]["device"]["device_id"], "camera:1")
        self.assertEqual(result["roles"]["audio.primary_output"]["settings"]["volume_percent"], 80)

    def test_required_disconnect_marks_profile_unhealthy(self):
        result = self.manager([]).resolve_all()
        self.assertFalse(result["healthy"])
        self.assertEqual(result["required_failures"], ["camera.front"])
        with self.assertRaises(ResourceNotFoundError):
            self.manager([]).resolve("camera.front")

    def test_ambiguous_match_is_not_silently_selected(self):
        devices = [
            {"device_id": f"camera:{n}", "kind": "camera", "stable_id": f"CAM_SERIAL-A_{n}", "name": "Front", "transport": "usb", "connected": True}
            for n in (1, 2)
        ]
        result = self.manager(devices).resolve_all()["roles"]["camera.front"]
        self.assertEqual(result["status"], "ambiguous")
        self.assertIsNone(result["device"])

    def test_rejects_unknown_selector_operator(self):
        self.profile["roles"]["camera.front"]["selector"] = {"regex": ".*"}
        with self.assertRaises(SchemaValidationError):
            _validate_profile(self.profile)


if __name__ == "__main__":
    unittest.main()
