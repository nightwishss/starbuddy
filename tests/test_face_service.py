import unittest
from pathlib import Path

from face_service.service import FaceService
from face_service.tracker import FaceTracker


def detection(x=0.1, score=0.9):
    return {"bbox": (x, 0.1, 0.2, 0.3), "score": score}


class FaceTrackerTests(unittest.TestCase):
    def test_requires_repeated_detection_before_entered_event(self):
        tracker = FaceTracker(confirm_hits=3, prefix="test")
        self.assertEqual(tracker.update([detection()], 0.0), [])
        track_id = tracker.snapshot(0.0)["tracks"][0]["track_id"]
        self.assertEqual(tracker.update([detection(0.11)], 0.2), [])
        events = tracker.update([detection(0.12)], 0.4)
        self.assertEqual(events[0]["type"], "entered")
        self.assertEqual(events[0]["track_id"], track_id)
        self.assertEqual(tracker.snapshot(0.4)["face_count"], 1)

    def test_confirmed_track_emits_left_and_is_removed(self):
        tracker = FaceTracker(confirm_hits=1, lost_after=1.0, prefix="test")
        entered = tracker.update([detection()], 0.0)
        self.assertEqual(entered[0]["type"], "entered")
        self.assertEqual(tracker.update([], 0.5), [])
        left = tracker.update([], 1.1)
        self.assertEqual(left[0]["type"], "left")
        self.assertEqual(tracker.snapshot(1.1)["tracks"], [])

    def test_separate_faces_receive_separate_ephemeral_tracks(self):
        tracker = FaceTracker(confirm_hits=1, prefix="test")
        tracker.update([detection(0.1), detection(0.65)], 0.0)
        ids = [track["track_id"] for track in tracker.snapshot(0.0)["tracks"]]
        self.assertEqual(ids, ["face_test_1", "face_test_2"])


class FakeWorker:
    def __init__(self, source, _detector, callback, fps):
        self.source, self.callback, self.fps = source, callback, fps
    def start(self): pass
    def close(self): pass


class FakeCameraClient:
    calls = []
    def __init__(self, **values):
        self.values = values
        self.calls.append(values)
    def wait_frame(self, after, timeout):
        return 12, b"\xff\xd8test\xff\xd9"


class FaceServiceTests(unittest.TestCase):
    def test_status_is_semantic_and_declares_no_identity_or_raw_retention(self):
        service = FaceService(["camera.front"], lambda _source: object(),
                              worker_factory=FakeWorker, wall_clock=lambda: 100.0)
        service._record("camera.front", 7, {
            "face_count": 1, "candidate_count": 0,
            "tracks": [{"track_id": "face_front_1", "state": "confirmed"}],
        }, None, [], latency_ms=25.0)
        status = service.status()
        self.assertTrue(status["ready"])
        self.assertFalse(status["identity_recognition"])
        self.assertFalse(status["privacy"]["raw_frames_retained"])
        self.assertEqual(status["sources"][0]["sequence"], 7)

        service._record("camera.front", None, {
            "face_count": 0, "candidate_count": 0, "tracks": [],
        }, "frame_unavailable", [])
        failed = service.status()["sources"][0]
        self.assertFalse(failed["ready"])
        self.assertEqual(failed["sequence"], 7)

    def test_debug_frame_is_on_demand_and_rejects_unknown_source(self):
        FakeCameraClient.calls.clear()
        service = FaceService(["camera.front"], lambda _source: object(),
                              worker_factory=FakeWorker,
                              camera_client_factory=FakeCameraClient)
        self.assertEqual(service.debug_frame("camera.front"),
                         (12, b"\xff\xd8test\xff\xd9"))
        self.assertEqual(FakeCameraClient.calls[0]["source"], "camera.front")
        self.assertTrue(service.status()["debug_preview"]["active"])
        self.assertEqual(service.status()["debug_preview"]["request_count"], 1)
        with self.assertRaises(KeyError):
            service.debug_frame("camera.unknown")

    def test_debug_ui_uses_external_assets_and_native_sequential_images(self):
        root = Path(__file__).parents[1] / "face_service" / "static"
        html = (root / "debug.html").read_text()
        script = (root / "app.js").read_text()
        self.assertIn('src="/debug/app.js"', html)
        self.assertNotIn("<script>", html)
        self.assertIn("camera.front", html)
        self.assertIn("camera.environment", html)
        self.assertIn("image.src = `/debug/frame.jpg", script)
        self.assertNotIn("createObjectURL", script)
        self.assertIn('document.addEventListener("visibilitychange"', script)


if __name__ == "__main__":
    unittest.main()
