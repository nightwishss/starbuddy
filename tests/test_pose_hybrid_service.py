#!/usr/bin/env python3
"""Focused checks for headless posture inference preview behavior."""

import time
import unittest

try:
    from pose_service.hybrid_service import PoseRuntime
except ModuleNotFoundError as exc:
    if exc.name != "cv2":
        raise
    PoseRuntime = None


@unittest.skipUnless(PoseRuntime is not None, "OpenCV pose runtime is not installed")
class PoseHybridServiceTests(unittest.TestCase):
    def test_background_inference_does_not_encode_preview_until_requested(self):
        runtime = PoseRuntime("camera", "worker", preview_ttl=.1)
        quality = {"upper_body_valid": True, "full_core_valid": False}

        runtime._publish(object(), None, False, quality, 5.0, {"confidence": .9})

        self.assertEqual(runtime.sequence, 1)
        self.assertIsNone(runtime.latest)
        self.assertFalse(runtime.status()["preview_active"])

    def test_frame_request_activates_bounded_preview_window(self):
        runtime = PoseRuntime("camera", "worker", preview_ttl=.1)

        self.assertIsNone(runtime.frame())
        self.assertTrue(runtime.status()["preview_active"])
        time.sleep(.12)
        self.assertFalse(runtime.status()["preview_active"])


if __name__ == "__main__":
    unittest.main()
