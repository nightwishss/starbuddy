#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock

from sdk.vision import VisionClient, VisionError


class VisionSDKTests(unittest.TestCase):
    def test_detects_only_owned_ephemeral_media(self):
        runtime = MagicMock()
        runtime.call.return_value = {"ok": True, "result": {"detections": []}}
        result = VisionClient(runtime).detect_objects(
            "image_test_media_object_1234", "session:test", confidence=0.45,
        )
        self.assertEqual(result["detections"], [])
        runtime.call.assert_called_once_with("vision.detect_objects", {
            "media_id": "image_test_media_object_1234",
            "media_owner": "session:test", "confidence": 0.45,
        }, timeout_ms=15_000)

    def test_runtime_failure_is_typed(self):
        runtime = MagicMock()
        runtime.call.return_value = {"ok": False, "error": {"code": "HARDWARE_ERROR"}}
        with self.assertRaises(VisionError):
            VisionClient(runtime).detect_objects("image_test_media_object_1234", "session:test")


if __name__ == "__main__":
    unittest.main()
