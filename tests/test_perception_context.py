import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path

from perception_context_service.client import PerceptionContextClient
from perception_context_service.context import ContextError, SemanticContext
from perception_context_service.providers import (
    ProviderPoller, face_observations, identity_observations, posture_observations,
)
from perception_context_service.service import ContextService, Handler, Server


class PerceptionContextTests(unittest.TestCase):
    def setUp(self):
        self.now = [1000.0]
        self.context = SemanticContext(clock=lambda: self.now[0])

    def observe(self, **changes):
        values = {
            "provider": "test-provider", "kind": "objects", "source": "camera.front",
            "observed_at": self.now[0], "ttl_seconds": 10, "confidence": .9,
            "inference": True,
            "data": {"counts": {"person": 1}, "confidences": {"person": .9},
                     "model": "test-model"},
        }
        values.update(changes)
        return self.context.observe(**values)

    def test_snapshot_is_semantic_bounded_and_expires(self):
        self.assertTrue(self.observe()["accepted"])
        snapshot = self.context.snapshot()
        self.assertEqual(snapshot["objects"][0]["label"], "person")
        self.assertFalse(snapshot["privacy"]["raw_media_present"])
        self.assertNotIn("data", snapshot["observations"][0])
        revision = snapshot["revision"]
        self.now[0] = 1010.0
        expired = self.context.snapshot()
        self.assertEqual(expired["objects"], [])
        self.assertFalse(expired["fresh"])
        self.assertGreater(expired["revision"], revision)

    def test_older_observation_does_not_replace_newer(self):
        self.observe(observed_at=999.0)
        result = self.observe(observed_at=998.0)
        self.assertFalse(result["accepted"])
        self.assertEqual(self.context.snapshot()["objects"][0]["observed_at"], 999.0)

    def test_identity_is_consent_gated_and_associated_only_to_one_same_source_face(self):
        self.context.observe(
            provider="face-yunet", kind="face_presence", source="camera.front",
            observed_at=1000.0, ttl_seconds=3, confidence=.91, inference=True,
            data={"tracks": [{"track_id": "face_front_1", "state": "confirmed",
                              "confidence": .91}]},
        )
        with self.assertRaisesRegex(ContextError, "consent"):
            self.context.observe(
                provider="face-sface", kind="identity", source="camera.front",
                observed_at=1000.0, ttl_seconds=3, confidence=.8, inference=True,
                data={"status": "known", "label": "April"},
            )
        self.context.observe(
            provider="face-sface", kind="identity", source="camera.front",
            observed_at=1000.0, ttl_seconds=3, confidence=.8, inference=True,
            data={"status": "known", "label": "April", "consent": "enrolled_local"},
        )
        person = self.context.snapshot()["people"][0]
        self.assertEqual(person["identity"]["label"], "April")
        self.assertEqual(person["identity"]["provenance"]["association"],
                         "single_face_same_source")
        self.assertEqual(self.context.snapshot()["identities"][0]["associated_track_id"],
                         "face_front_1")

    def test_ambiguous_identity_cue_is_not_bound_to_a_person(self):
        self.context.observe(
            provider="face-yunet", kind="face_presence", source="camera.front",
            observed_at=1000.0, ttl_seconds=3, confidence=.9, inference=True,
            data={"tracks": [
                {"track_id": "face_front_1", "state": "confirmed", "confidence": .9},
                {"track_id": "face_front_2", "state": "confirmed", "confidence": .85},
            ]},
        )
        self.context.observe(
            provider="face-sface", kind="identity", source="camera.front",
            observed_at=1000.0, ttl_seconds=3, confidence=.8, inference=True,
            data={"status": "known", "label": "April", "consent": "enrolled_local"},
        )
        snapshot = self.context.snapshot()
        self.assertTrue(all(item["identity"] is None for item in snapshot["people"]))
        self.assertIsNone(snapshot["identities"][0]["associated_track_id"])

    def test_raw_media_and_embeddings_are_not_valid_fields(self):
        with self.assertRaises(ContextError):
            self.context.observe(
                provider="test", kind="objects", source="camera.front", observed_at=1000,
                ttl_seconds=10, inference=True,
                data={"counts": {"person": 1}, "confidences": {}, "jpeg": "secret"},
            )

    def test_posture_retains_observation_and_inference_metadata(self):
        self.context.observe(
            provider="posture-policy", kind="posture", source="camera.environment",
            observed_at=1000, ttl_seconds=3, confidence=.95, inference=True,
            data={"state": "suspect", "reasons": ["head_forward"],
                  "enabled": True, "calibrated": True},
        )
        posture = self.context.snapshot()["posture"]
        self.assertEqual(posture["state"], "suspect")
        self.assertEqual(posture["confidence"], .95)
        self.assertTrue(posture["provenance"]["inference"])

    def test_object_labels_support_bounded_coco_names_with_spaces(self):
        self.observe(data={
            "counts": {"traffic light": 1}, "confidences": {"traffic light": .81},
            "model": "YOLOX-Nano-QNN-416",
        })
        self.assertEqual(self.context.snapshot()["objects"][0]["label"], "traffic light")

    def test_affect_is_ephemeral_and_bound_to_one_confirmed_track(self):
        labels = ("neutral", "happy_like", "surprised_like", "sad_like", "angry_like",
                  "disgust_like", "fear_like", "contempt_like")
        scores = {label: (.79 if label == "happy_like" else .03) for label in labels}
        self.context.observe(provider="face-yunet", kind="face_presence", source="camera.front",
                             observed_at=1000, ttl_seconds=5, confidence=.9, inference=True,
                             data={"tracks": [{"track_id": "face_front_1", "state": "confirmed",
                                               "confidence": .9}]})
        values = dict(provider="ferplus-cpu", kind="affect_cue", source="camera.front",
                      observed_at=1000, ttl_seconds=3, confidence=.79, inference=True,
                      data={"track_id": "face_front_1", "top_label": "happy_like",
                            "confidence": .79, "distribution": scores,
                            "model": "emotion-ferplus-8"})
        self.context.observe(**values)
        cue = self.context.snapshot()["affect_cues"][0]
        self.assertEqual(cue["track_id"], "face_front_1")
        self.assertEqual(cue["top_label"], "happy_like")
        self.assertEqual(cue["provenance"]["association"], "confirmed_single_face_track")
        self.now[0] = 1003
        self.assertEqual(self.context.snapshot()["affect_cues"], [])

    def test_affect_rejects_raw_fields_and_ambiguous_track(self):
        scores = {label: (.79 if label == "neutral" else .03) for label in (
            "neutral", "happy_like", "surprised_like", "sad_like", "angry_like",
            "disgust_like", "fear_like", "contempt_like")}
        data = {"track_id": "face_front_1", "top_label": "neutral",
                "confidence": .79, "distribution": scores, "model": "emotion-ferplus-8"}
        with self.assertRaises(ContextError):
            self.context.observe(provider="ferplus-cpu", kind="affect_cue",
                                 source="camera.front", observed_at=1000, ttl_seconds=3,
                                 confidence=.79, inference=True, data={**data, "jpeg": "secret"})
        self.context.observe(provider="ferplus-cpu", kind="affect_cue",
                             source="camera.front", observed_at=1000, ttl_seconds=3,
                             confidence=.79, inference=True, data=data)
        self.assertEqual(self.context.snapshot()["affect_cues"], [])


class ProviderAdapterTests(unittest.TestCase):
    def test_existing_status_shapes_become_bounded_observations(self):
        face = face_observations({"sources": [{
            "source": "camera.front", "ready": True, "last_frame_at": 99.5,
            "tracks": [{"track_id": "face_front_1", "state": "confirmed", "score": .9,
                        "bbox": [0, 0, 1, 1]}],
        }]}, 100.0)[0]
        self.assertNotIn("bbox", face["data"]["tracks"][0])
        identity = identity_observations({
            "ready": True, "source": "camera.front", "last_frame_seconds_ago": .2,
            "observation": {"status": "known", "label": "April", "similarity": .8,
                            "identity_id": "private"},
        }, 100.0)[0]
        self.assertEqual(identity["data"], {
            "status": "known", "label": "April", "consent": "enrolled_local",
        })
        posture = posture_observations({
            "state": "healthy", "reasons": [], "enabled": True, "calibrated": True,
            "pose": {"posture_features": {"confidence": .94, "values": {"private": 1}}},
            "camera": {"role": "camera.environment", "endpoint": "/dev/video4"},
        }, 100.0)[0]
        self.assertEqual(posture["confidence"], .94)
        self.assertNotIn("values", posture["data"])

    def test_provider_failure_is_isolated(self):
        context = SemanticContext(clock=lambda: 100.0)
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): pass
            def read(self): return b'{"sources":[]}'
        def opener(url, **_kwargs):
            if "bad" in url: raise OSError("down")
            return Response()
        poller = ProviderPoller(
            context, endpoints={"face": "http://ok", "identity": "http://bad"},
            clock=lambda: 100.0, opener=opener,
        )
        poller.poll_once()
        status = poller.status()
        self.assertTrue(status["face"]["ready"])
        self.assertFalse(status["identity"]["ready"])
        self.assertEqual(status["identity"]["last_error"], "OSError")


class PerceptionContextServiceTests(unittest.TestCase):
    def test_private_socket_observe_snapshot_and_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.sock"
            context = SemanticContext()
            server = Server(str(path), Handler)
            server.service = ContextService(context)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                client = PerceptionContextClient(str(path))
                client.observe(
                    provider="yolox", kind="objects", source="camera.environment",
                    observed_at=context.clock(), ttl_seconds=10, confidence=.8, inference=True,
                    data={"counts": {"cup": 1}, "confidences": {"cup": .8},
                          "model": "yolox-nano"},
                )
                self.assertEqual(client.snapshot()["objects"][0]["label"], "cup")
                self.assertEqual(client.stats()["active_observations"], 1)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
