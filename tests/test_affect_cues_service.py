import unittest

from affect_cues_service.engine import AffectWorker, CueSmoother, LABELS, _softmax
from perception_context_service.context import SemanticContext


def distribution(label, confidence=.82):
    rest = (1 - confidence) / (len(LABELS) - 1)
    return {item: confidence if item == label else rest for item in LABELS}


class FakeCamera:
    def __init__(self):
        self.sequence = 0

    def wait_frame(self, after_sequence, timeout):
        self.sequence += 1
        return self.sequence, b"camera-frame"


class FakeModel:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def infer(self, frame, bbox, *, rotation):
        self.calls += 1
        assert frame == b"camera-frame"
        assert rotation == 180
        return self.value


class FakeClient:
    def __init__(self, context):
        self.context = context

    def observe(self, **kwargs):
        return self.context.observe(**kwargs)


class FakeResources:
    def __init__(self, *, deny=False):
        self.deny = deny
        self.acquired = []
        self.released = []

    def acquire(self, owner, resources, purpose, **options):
        if self.deny:
            raise RuntimeError("resource busy")
        self.acquired.append((owner, resources, purpose, options))
        return {"lease_id": "lease_test"}

    def release(self, lease_id, owner):
        self.released.append((lease_id, owner))


class AffectCueTests(unittest.TestCase):
    def test_softmax_rejects_nonfinite_output(self):
        self.assertAlmostEqual(sum(_softmax([0] * 8).values()), 1, places=3)
        with self.assertRaises(ValueError):
            _softmax([float("nan")] * 8)

    def test_smoother_needs_consistent_confident_samples_and_resets_track(self):
        smoother = CueSmoother()
        for _ in range(2):
            self.assertIsNone(smoother.update("face_front_1", distribution("happy_like")))
        self.assertEqual(smoother.update("face_front_1", distribution("happy_like"))["top_label"],
                         "happy_like")
        self.assertIsNone(smoother.update("face_front_2", distribution("happy_like")))
        self.assertIsNone(CueSmoother(required=3).update("face_front_1",
                                                     distribution("sad_like", .35)))

    def test_worker_uses_one_shared_camera_and_submits_semantic_only(self):
        now = [1000.0]
        context = SemanticContext(clock=lambda: now[0])
        context.observe(provider="face-yunet", kind="face_presence", source="camera.front",
                        observed_at=1000.0, ttl_seconds=30, confidence=.9, inference=True,
                        data={"tracks": [{"track_id": "face_front_1", "state": "confirmed",
                                          "confidence": .9}]})
        status = lambda: {"sources": [{"source": "camera.front", "ready": True,
                    "last_frame_at": now[0], "tracks": [{"track_id": "face_front_1",
                    "state": "confirmed", "last_seen_seconds_ago": 0,
                    "bbox": [.2, .2, .4, .4]}]}]}
        model = FakeModel(distribution("happy_like"))
        worker = AffectWorker(model, FakeCamera(), status, FakeClient(context),
                              clock=lambda: now[0], monotonic=lambda: 1)
        for _ in range(3):
            worker.tick()
        cue = context.snapshot()["affect_cues"][0]
        self.assertEqual(cue["top_label"], "happy_like")
        self.assertEqual(cue["track_id"], "face_front_1")
        self.assertNotIn("bbox", cue)
        self.assertNotIn("jpeg", str(context.snapshot()))
        now[0] += 3
        self.assertEqual(context.snapshot()["affect_cues"], [])

    def test_multiple_faces_prevent_inference(self):
        model = FakeModel(distribution("happy_like"))
        status = lambda: {"sources": [{"source": "camera.front", "ready": True,
                    "last_frame_at": 1000, "tracks": [
                        {"track_id": name, "state": "confirmed", "last_seen_seconds_ago": 0,
                         "bbox": [.2, .2, .4, .4]} for name in ("one", "two")]}]}
        worker = AffectWorker(model, FakeCamera(), status, FakeClient(SemanticContext()),
                              clock=lambda: 1000)
        worker.tick()
        self.assertEqual(model.calls, 0)

    def test_opt_in_resource_denial_skips_best_effort_inference(self):
        model = FakeModel(distribution("happy_like"))
        resources = FakeResources(deny=True)
        status = lambda: {"sources": [{"source": "camera.front", "ready": True,
                    "last_frame_at": 1000, "tracks": [{"track_id": "one",
                    "state": "confirmed", "last_seen_seconds_ago": 0,
                    "bbox": [.2, .2, .4, .4]}]}]}
        worker = AffectWorker(model, FakeCamera(), status, FakeClient(SemanticContext()),
                              clock=lambda: 1000, resources=resources)
        worker.tick()
        self.assertEqual(model.calls, 0)
        self.assertEqual(worker.status()["resource_skips"], 1)
        self.assertEqual(worker.status()["last_error"], "resource_unavailable")
        resources.deny = False
        worker.tick()
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(resources.acquired), 1)
        self.assertEqual(resources.released, [("lease_test", "model:affect-ferplus")])
        self.assertEqual(resources.acquired[0][1], {"cpu_compute": 10})
        self.assertEqual(resources.acquired[0][3]["priority"], 10)

    def test_resource_lease_released_when_model_fails(self):
        class BrokenModel:
            def infer(self, *_args, **_kwargs):
                raise ValueError("model failed")
        resources = FakeResources()
        status = lambda: {"sources": [{"source": "camera.front", "ready": True,
                    "last_frame_at": 1000, "tracks": [{"track_id": "one",
                    "state": "confirmed", "last_seen_seconds_ago": 0,
                    "bbox": [.2, .2, .4, .4]}]}]}
        worker = AffectWorker(BrokenModel(), FakeCamera(), status, FakeClient(SemanticContext()),
                              clock=lambda: 1000, resources=resources)
        worker.tick()
        self.assertEqual(resources.released, [("lease_test", "model:affect-ferplus")])
        self.assertEqual(worker.status()["last_error"], "ValueError")


if __name__ == "__main__":
    unittest.main()
