#!/usr/bin/env python3
"""Composition and privacy tests for structured observations."""

import json
import unittest
from unittest.mock import MagicMock, patch

from capability.perception import impl


FRAME = {
    "frame_id": "frame_ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "media_id": "image_ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "camera_id": "camera-abc", "width": 640, "height": 360,
}


class PerceptionCapabilityTests(unittest.TestCase):
    def setUp(self):
        impl._PREVIOUS.clear()

    @patch.object(impl.camera, "release_frame")
    @patch.object(impl.vision, "detect_objects")
    @patch.object(impl.camera, "snapshot", return_value=FRAME)
    def test_observe_composes_and_releases_pixels(self, snapshot, detect, release):
        detect.return_value = {
            "detections": [
                {"label": "person", "confidence": 0.91, "x": 1, "y": 2, "width": 3, "height": 4},
                {"label": "cup", "confidence": 0.80, "x": 5, "y": 6, "width": 7, "height": 8},
            ],
            "model": "YOLOX-Nano-QNN-416", "backend": "QNN/HTP", "timing": {},
        }

        result = impl.observe({"role": "camera.front", "profile": "fast", "confidence": 0.4})

        self.assertTrue(result["summary"]["human_present"])
        self.assertEqual(result["summary"]["counts"], {"cup": 1, "person": 1})
        self.assertEqual(result["stable_summary"]["counts"], {"cup": 1, "person": 1})
        self.assertTrue(result["change"]["baseline"])
        self.assertNotIn("media_id", result)
        snapshot.assert_called_once_with({
            "role": "camera.front", "transform": {"resize": {"width": 640, "height": 360}},
            "quality": 72,
        })
        detect.assert_called_once_with({
            "media_id": FRAME["media_id"], "media_owner": "service:camera-runtime",
            "confidence": 0.4,
        })
        release.assert_called_once_with({"frame_id": FRAME["frame_id"]})

    @patch.object(impl.camera, "release_frame")
    @patch.object(impl.vision, "detect_objects")
    @patch.object(impl.camera, "snapshot", return_value=FRAME)
    def test_change_reports_added_and_removed_counts(self, _snapshot, detect, _release):
        detect.return_value = {
            "detections": [{"label": "person", "confidence": 0.9}],
            "model": "model", "backend": "QNN/HTP", "timing": {},
        }
        impl.observe({})
        detect.return_value["detections"] = [{"label": "cup", "confidence": 0.8}]

        provisional = impl.observe({})
        result = impl.observe({})

        self.assertTrue(provisional["change"]["provisional"])
        self.assertFalse(provisional["change"]["changed"])
        self.assertEqual(provisional["stable_summary"]["counts"], {"person": 1})
        self.assertTrue(result["change"]["changed"])
        self.assertEqual(result["change"]["added"], {"cup": 1})
        self.assertEqual(result["change"]["removed"], {"person": 1})

    @patch.object(impl.camera, "release_frame")
    @patch.object(impl.vision, "detect_objects")
    @patch.object(impl.camera, "snapshot", return_value=FRAME)
    def test_single_frame_flicker_does_not_change_stable_scene(self, _snapshot, detect, _release):
        detect.return_value = {
            "detections": [{"label": "person", "confidence": 0.9}],
            "model": "model", "backend": "QNN/HTP", "timing": {},
        }
        impl.observe({})
        detect.return_value["detections"] = [{"label": "cup", "confidence": 0.8}]
        flicker = impl.observe({})
        detect.return_value["detections"] = [{"label": "person", "confidence": 0.9}]
        recovered = impl.observe({})

        self.assertTrue(flicker["change"]["provisional"])
        self.assertEqual(flicker["stable_summary"]["counts"], {"person": 1})
        self.assertFalse(recovered["change"]["changed"])
        self.assertTrue(recovered["change"]["stable"])
        self.assertEqual(recovered["stable_summary"]["counts"], {"person": 1})

    @patch.object(impl.camera, "release_frame")
    @patch.object(impl.vision, "detect_objects", side_effect=RuntimeError("model down"))
    @patch.object(impl.camera, "snapshot", return_value=FRAME)
    def test_frame_is_released_when_inference_fails(self, _snapshot, _detect, release):
        with self.assertRaises(RuntimeError):
            impl.observe({})
        release.assert_called_once_with({"frame_id": FRAME["frame_id"]})

    @patch.object(impl.urllib.request, "urlopen")
    def test_posture_status_is_pixel_free_loopback_adapter(self, urlopen):
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"enabled":true,"state":"healthy"}'
        urlopen.return_value = response

        result = impl.ACTIONS["perception.posture.status"]({})

        self.assertEqual(result, {"enabled": True, "state": "healthy"})
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8792/api/posture/status")
        self.assertEqual(request.method, "GET")

    @patch.object(impl.urllib.request, "urlopen")
    def test_posture_monitor_forwards_only_bounded_json(self, urlopen):
        response=MagicMock(); response.__enter__.return_value=response
        response.read.return_value=b'{"accepted":true}' ; urlopen.return_value=response
        arguments={"task_id":"task_"+"a"*32,"duration_seconds":600,
                   "bad_duration_seconds":20,"cooldown_seconds":180}
        self.assertTrue(impl.ACTIONS["perception.posture.monitor"](arguments)["accepted"])
        request=urlopen.call_args.args[0]
        self.assertEqual(request.full_url,"http://127.0.0.1:8792/api/posture/monitor")
        self.assertEqual(json.loads(request.data),arguments)

    @patch.object(impl.urllib.request, "urlopen")
    def test_posture_stop_forwards_only_task_identity(self, urlopen):
        response=MagicMock(); response.__enter__.return_value=response
        response.read.return_value=b'{"stopped":true}' ; urlopen.return_value=response
        task_id="task_"+"a"*32
        self.assertTrue(impl.ACTIONS["perception.posture.stop"]({"task_id":task_id})["stopped"])
        request=urlopen.call_args.args[0]
        self.assertEqual(request.full_url,"http://127.0.0.1:8792/api/posture/stop")
        self.assertEqual(json.loads(request.data),{"task_id":task_id})

    @patch.object(impl._CONTEXT, "snapshot")
    def test_context_snapshot_uses_semantic_owner_service(self, snapshot):
        snapshot.return_value = {
            "revision": 1, "generated_at": 1000.0, "people": [], "identities": [], "objects": [],
            "posture": None, "observations": [], "fresh": False,
            "privacy": {"raw_media_present": False},
        }
        result = impl.ACTIONS["perception.context.snapshot"]({})
        self.assertFalse(result["privacy"]["raw_media_present"])
        snapshot.assert_called_once_with()
        with self.assertRaises(ValueError):
            impl.ACTIONS["perception.context.snapshot"]({"raw": True})


if __name__ == "__main__":
    unittest.main()
