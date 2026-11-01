#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, call

from sdk.device import DeviceClient, DeviceError


class DeviceSdkTests(unittest.TestCase):
    def test_profile_inventory_status_and_resolve_are_bounded(self):
        runtime = MagicMock()
        runtime.call.side_effect = [
            {"ok": True, "result": {"profile_id": "test"}},
            {"ok": True, "result": {"devices": [], "count": 0}},
            {"ok": True, "result": {"healthy": True, "roles": {}}},
            {"ok": True, "result": {"role": "camera.front", "status": "resolved"}},
        ]
        client = DeviceClient(runtime)
        client.profile(); client.inventory(); client.status(); client.resolve("camera.front")
        self.assertEqual(runtime.call.call_args_list, [
            call("device.profile", {}, timeout_ms=1_000),
            call("device.inventory", {}, timeout_ms=6_000),
            call("device.status", {}, timeout_ms=6_000),
            call("device.resolve", {"role": "camera.front"}, timeout_ms=6_000),
        ])

    def test_structured_error_and_local_role_validation(self):
        runtime = MagicMock(return_value=None)
        runtime.call.return_value = {
            "ok": False, "error": {"code": "RESOURCE_NOT_FOUND", "message": "missing"},
        }
        with self.assertRaises(DeviceError) as raised:
            DeviceClient(runtime).status()
        self.assertEqual(raised.exception.code, "RESOURCE_NOT_FOUND")
        with self.assertRaises(ValueError):
            DeviceClient(runtime).resolve("")
        with self.assertRaises(ValueError):
            DeviceClient(runtime).status(timeout_ms=6_001)

    def test_transport_details_are_wrapped(self):
        runtime = MagicMock()
        runtime.call.side_effect = OSError("private socket path")
        with self.assertRaises(DeviceError) as raised:
            DeviceClient(runtime).inventory()
        self.assertEqual(raised.exception.code, "TRANSPORT_ERROR")
        self.assertNotIn("socket", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
