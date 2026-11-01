#!/usr/bin/env python3

import unittest
from unittest.mock import patch

from camera_stream_service.client import CameraStreamClient
from camera_stream_service.service import CameraBroker, CameraStream


class FakeProfileManager:
    def profile(self):
        return {"roles": {
            "camera.front": {"kind": "camera"},
            "camera.environment": {"kind": "camera"},
            "audio.primary_input": {"kind": "audio_input"},
        }}

    def resolve_all(self):
        return {"roles": {
            "camera.front": {"status": "resolved", "device": {
                "device_id": "camera:front", "stable_id": "usb-front", "endpoint": "/dev/video4",
            }},
            "camera.environment": {"status": "missing", "device": None},
        }}


class FakeStream:
    def __init__(self, _manager, role):
        self.role = role
        self.started = False
        self.closed = False
        self.source_id = None
        self.stable_id = None

    def update_resolution(self, resolved):
        device = resolved.get("device") if resolved.get("status") == "resolved" else None
        self.source_id = device.get("device_id") if device else None
        self.stable_id = device.get("stable_id") if device else None

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def status(self):
        return {
            "ready": self.role == "camera.front",
            "role": self.role,
            "connected": self.source_id is not None,
            "source_id": self.source_id,
            "stable_id": self.stable_id,
        }


class CameraStreamTests(unittest.TestCase):
    def test_latest_frame_replaces_old_data_and_counts_source(self):
        stream = CameraStream(profile_manager=object())
        stream.latest = b"old"
        stream.sequence = 1
        # Exercise the bounded state contract without starting hardware.
        with stream.condition:
            stream.latest = b"new"
            stream.sequence += 1
            stream.source_frames += 1
        self.assertEqual(stream.wait_frame(1, timeout=0), (2, b"new", None))
        self.assertEqual(stream.status()["queue_policy"], "latest-frame-drop-old")

    def test_consumer_metrics_account_for_dropped_intermediate_frames(self):
        stream = CameraStream(profile_manager=object())
        token = stream.register("pose", 10)
        stream.account(token, 4, 9)
        status = stream.status()
        self.assertEqual(status["subscribers"][0]["sent"], 1)
        self.assertEqual(status["subscribers"][0]["dropped"], 4)
        stream.unregister(token)
        self.assertEqual(stream.status()["dropped_frames"], 4)

    def test_source_is_demand_driven(self):
        stream = CameraStream(profile_manager=object())
        with stream.condition:
            self.assertFalse(stream._needed_locked())
        token = stream.register("display", 6)
        with stream.condition:
            self.assertTrue(stream._needed_locked())
        stream.unregister(token)

    def test_stalled_pipe_sets_error_and_forces_restart(self):
        stream = CameraStream(profile_manager=object())
        stream.request_frame()
        with patch("camera_stream_service.service.select.select", return_value=([], [], [])):
            with self.assertRaises(TimeoutError):
                stream._read_frames(object())
        self.assertEqual(stream.status()["error"], "frame_stale")

    def test_client_adds_optional_source_without_breaking_legacy_url(self):
        legacy = CameraStreamClient()._url("/status")
        front = CameraStreamClient(source="camera.front")._url(
            "/frame.jpg", "consumer=face&after=7")
        self.assertEqual(legacy, "http://127.0.0.1:8781/status")
        self.assertEqual(
            front,
            "http://127.0.0.1:8781/frame.jpg?consumer=face&after=7&source=camera.front",
        )


class CameraBrokerTests(unittest.TestCase):
    def test_registers_every_camera_role_and_preserves_environment_default(self):
        broker = CameraBroker(FakeProfileManager(), stream_factory=FakeStream)
        self.assertEqual(set(broker.streams), {"camera.front", "camera.environment"})
        self.assertEqual(broker.default_role, "camera.environment")
        self.assertEqual(broker.select().role, "camera.environment")

    def test_selects_source_by_role_device_id_or_stable_id(self):
        broker = CameraBroker(FakeProfileManager(), stream_factory=FakeStream)
        self.assertEqual(broker.select("camera.front").role, "camera.front")
        self.assertEqual(broker.select("camera:front").role, "camera.front")
        self.assertEqual(broker.select("usb-front").role, "camera.front")
        self.assertIsNone(broker.select("missing"))

    def test_aggregate_status_reports_configured_and_ready_sources(self):
        broker = CameraBroker(FakeProfileManager(), stream_factory=FakeStream)
        status = broker.status()
        self.assertTrue(status["ready"])
        self.assertEqual(status["source_count"], 2)
        self.assertEqual(status["connected_source_count"], 1)
        self.assertEqual(status["ready_source_count"], 1)
        self.assertEqual(status["ownership"], "single-owner-per-physical-source")


if __name__ == "__main__":
    unittest.main()
