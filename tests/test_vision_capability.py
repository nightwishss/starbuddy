#!/usr/bin/env python3
"""Boundary and result-normalization tests for vision capabilities."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from capability.vision import impl
from runtime.errors import ResourceNotFoundError, SchemaValidationError


class VisionCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.media_id = "image_ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        (self.root / f"{self.media_id}.jpg").write_bytes(b"\xff\xd8pixels\xff\xd9")
        self.root_patch = patch.object(impl, "_MEDIA_ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def test_resolves_only_opaque_image_ids(self):
        self.assertTrue(impl._image_source(self.media_id).is_file())
        with self.assertRaises(SchemaValidationError):
            impl._image_source("../../etc/passwd")
        with self.assertRaises(ResourceNotFoundError):
            impl._image_source("image_ZYXWVUTSRQPONMLKJIHGFEDCBA")

    @patch.object(impl, "_media_client")
    def test_resolves_shared_camera_media_with_fixed_owner(self, media_client):
        media_client.return_value.call.return_value = {
            "path": str(self.root / f"{self.media_id}.jpg")}
        path = impl._image_source(self.media_id, "service:camera-runtime")
        self.assertTrue(path.is_file())
        media_client.return_value.call.assert_called_once_with(
            "resolve", owner="service:camera-runtime", media_id=self.media_id)
        impl._image_source(self.media_id, "phone-session_123")
        with self.assertRaises(SchemaValidationError):
            impl._image_source(self.media_id, "invalid owner")

    @patch("capability.vision.impl._post_image")
    def test_normalizes_and_filters_runner_result(self, post):
        post.return_value = {
            "result": {
                "bounding_boxes": [
                    {"label": "person", "value": 0.9, "x": 1, "y": 2, "width": 3, "height": 4},
                    {"label": "cup", "value": 0.4, "x": 5, "y": 6, "width": 7, "height": 8},
                ]
            },
            "timing": {"classification": 3, "dsp": 1},
        }
        result = impl.detect_objects({"media_id": self.media_id, "confidence": 0.5})
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["detections"][0]["label"], "person")
        self.assertEqual(result["backend"], "QNN/HTP")
        self.assertEqual(result["timing"]["dsp"], 1.0)

    @patch("capability.vision.impl._post_image")
    def test_caps_untrusted_runner_results(self, post):
        post.return_value = {
            "result": {"bounding_boxes": [
                {"label": "x" * 500, "value": 1, "x": float("nan")} for _ in range(150)
            ]}
        }
        result = impl.detect_objects({"media_id": self.media_id})
        self.assertEqual(result["count"], 100)
        self.assertEqual(len(result["detections"][0]["label"]), 100)
        self.assertEqual(result["detections"][0]["x"], 0.0)


if __name__ == "__main__":
    unittest.main()
