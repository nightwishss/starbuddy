#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, call

from sdk.camera import CameraClient, CameraError


class CameraSdkTests(unittest.TestCase):
    def test_role_snapshot_maps_source_transform_and_trace(self):
        runtime = MagicMock()
        runtime.call.return_value = {
            "ok": True, "result": {"frame_id": "frame_test", "media_id": "image_test"},
        }
        result = CameraClient(runtime).snapshot(
            role="camera.environment",
            source={"width": 1280, "height": 720, "pixel_format": "mjpeg", "fps": 30},
            transform={"crop": {"x": 0, "y": 0, "width": 640, "height": 480},
                       "rotation": 90, "resize": {"width": 320, "height": 320}},
            quality=80, request_id="camera-sdk-test",
        )
        self.assertEqual(result["frame_id"], "frame_test")
        runtime.call.assert_called_once_with(
            "camera.snapshot", {
                "role": "camera.environment",
                "source": {"width": 1280, "height": 720, "pixel_format": "mjpeg", "fps": 30},
                "transform": {"crop": {"x": 0, "y": 0, "width": 640, "height": 480},
                              "rotation": 90, "resize": {"width": 320, "height": 320}},
                "quality": 80,
            }, timeout_ms=15_000, request_id="camera-sdk-test",
        )

    def test_frame_context_always_releases(self):
        runtime = MagicMock()
        runtime.call.side_effect = [
            {"ok": True, "result": {"frame_id": "frame_test"}},
            {"ok": True, "result": {"frame_id": "frame_test", "released": True}},
        ]
        with self.assertRaisesRegex(RuntimeError, "consumer failed"):
            with CameraClient(runtime).frame(role="camera.front"):
                raise RuntimeError("consumer failed")
        self.assertEqual(runtime.call.call_args_list[-1], call(
            "camera.release_frame", {"frame_id": "frame_test"}, timeout_ms=1_000
        ))

    def test_exactly_one_selector_and_structured_error(self):
        with self.assertRaises(ValueError):
            CameraClient(MagicMock()).snapshot()
        with self.assertRaises(ValueError):
            CameraClient(MagicMock()).snapshot(camera_id="one", role="camera.front")
        runtime = MagicMock()
        runtime.call.return_value = {"ok": False, "error": {"code": "HARDWARE_ERROR", "message": "down"}}
        with self.assertRaises(CameraError):
            CameraClient(runtime).list()


if __name__ == "__main__":
    unittest.main()
