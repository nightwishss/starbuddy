#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock

from sdk.perception import PerceptionClient, PerceptionError


class PerceptionSDKTests(unittest.TestCase):
    def test_uses_hidden_composite_capability(self):
        runtime = MagicMock()
        runtime.call.return_value = {"ok": True, "result": {"summary": {"labels": []}}}
        result = PerceptionClient(runtime).observe(role="camera.front", profile="balanced", confidence=0.5)
        self.assertEqual(result["summary"]["labels"], [])
        runtime.call.assert_called_once_with("perception.observe", {
            "role": "camera.front", "profile": "balanced", "confidence": 0.5,
            "stability_samples": 2,
        }, timeout_ms=30_000)

    def test_public_error_does_not_require_internal_details(self):
        runtime = MagicMock()
        runtime.call.return_value = {"ok": False, "error": {"message": "camera unavailable"}}
        with self.assertRaises(PerceptionError):
            PerceptionClient(runtime).observe()

    def test_posture_methods_use_formal_runtime_capabilities(self):
        runtime = MagicMock()
        runtime.call.return_value = {"ok": True, "result": {"enabled": True, "state": "healthy"}}
        client = PerceptionClient(runtime)

        self.assertEqual(client.posture_status()["state"], "healthy")
        runtime.call.assert_called_with("perception.posture.status", {}, timeout_ms=3_000)
        client.posture_pause()
        runtime.call.assert_called_with("perception.posture.pause", {}, timeout_ms=3_000)


if __name__ == "__main__":
    unittest.main()
