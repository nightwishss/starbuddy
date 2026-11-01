"""Fresh, consented, media-free evidence for one interaction turn."""

import unittest

from interaction_context import assemble_turn_context


def observation(*, now=1000, source="camera.front", confidence=.9, **extra):
    return {"source": source, "confidence": confidence,
            "observed_at": now, "expires_at": now + 3,
            "provenance": {"provider": "test-provider", "inference": True}, **extra}


def fused(kind, *, source="camera.front", subject=None, state="present", now=1000,
          reference="current-evidence", **extra):
    return {"kind": kind, "source": source, "subject": subject, "session_id": None,
            "state": state, "observed_at": now, "arrived_at": now, "expires_at": now + 3,
            "evidence": [reference], **extra}


def identity_cue(label, track_id, *, now=1000, source="camera.front"):
    return {"label": label, "consent": "enrolled_local", "source": source,
            "associated_track_id": track_id, "observed_at": now, "expires_at": now + 3}


class TurnContextTests(unittest.TestCase):
    def test_fuses_only_fresh_matching_semantic_evidence(self):
        person = observation(track_id="track-1", identity=identity_cue("April", "track-1"))
        snapshot = {
            "revision": 7, "people": [person],
            "objects": [observation(source="camera.environment", label="cup", count=1)],
            "posture": observation(source="camera.environment", state="healthy", reasons=[]),
            "affect_cues": [observation(track_id="track-1", top_label="neutral", confidence=.82)],
            "observations": [], "raw_frame": "never copy this",
        }
        request = {"messages": [{"role": "user", "text": "你好"},
                                {"role": "assistant", "text": "你好"}]}
        result = assemble_turn_context(request, snapshot,
                                       {"status": "known", "label": "April", "current_verified": True},
                                       [{"state": "scheduled", "summary": "喝水提醒"}], now=1001)
        self.assertTrue(result["identity_verified"])
        self.assertEqual(result["evidence"]["conversation"]["turns"], 1)
        self.assertEqual(result["evidence"]["affect"]["provider"], "test-provider")
        self.assertEqual(result["tasks"], [{"state": "scheduled", "summary": "喝水提醒"}])
        self.assertEqual(result["perception"]["objects"][0]["label"], "cup")
        self.assertNotIn("raw_frame", result["perception"])
        self.assertNotIn("April", str(result["evidence"]))

    def test_expired_or_mismatched_identity_cannot_name_user(self):
        person = observation(track_id="track-1", identity=identity_cue("SomeoneElse", "track-1"))
        snapshot = {"people": [person], "objects": [], "affect_cues": [],
                    "posture": None, "observations": []}
        result = assemble_turn_context({}, snapshot,
                                       {"status": "known", "label": "April", "current_verified": True}, now=1001)
        self.assertFalse(result["identity_verified"])
        self.assertIsNone(result["perception"]["people"][0]["identity"])
        expired = assemble_turn_context({}, snapshot,
                                        {"status": "known", "label": "SomeoneElse",
                                         "current_verified": True}, now=1004)
        self.assertEqual(expired["evidence"]["identity"]["state"], "unconfirmed")
        self.assertEqual(expired["perception"]["people"], [])

    def test_unknown_voice_speaker_is_not_assigned_visible_face_identity(self):
        snapshot = {"people": [observation(track_id="front-1", identity=identity_cue("April", "front-1"))]}
        result = assemble_turn_context({"input_type": "voice"}, snapshot,
            {"status": "known", "label": "April", "current_verified": True}, now=1001)
        self.assertFalse(result["identity_verified"])
        self.assertIsNone(result["perception"]["people"][0]["identity"])

    def test_affect_requires_one_matching_fresh_front_track(self):
        person = observation(track_id="track-1")
        cue = observation(track_id="track-2", top_label="sad_like", confidence=.9)
        snapshot = {"people": [person], "affect_cues": [cue]}
        result = assemble_turn_context({}, snapshot, now=1001)
        self.assertEqual(result["perception"]["affect_cues"], [])
        snapshot["affect_cues"] = [observation(track_id="track-1", top_label="sad_like",
                                                confidence=.7)]
        result = assemble_turn_context({}, snapshot, now=1001)
        self.assertEqual(result["evidence"]["affect"]["state"], "uncertain")

    def test_absent_provider_is_explicitly_missing_not_imagined(self):
        result = assemble_turn_context({"messages": []}, None, None, None, now=1000)
        self.assertEqual(result["evidence"]["people"]["state"], "missing")
        self.assertEqual(result["evidence"]["tasks"]["state"], "unavailable")
        self.assertEqual(result["evidence"]["identity"]["state"], "unconfirmed")

    def test_departed_track_does_not_filter_current_affect_or_pollute_refs(self):
        snapshot = {"people": [observation(track_id="current")],
                    "affect_cues": [observation(track_id="current", top_label="neutral")],
                    "fusion": {"ready": True, "items": [
                        fused("presence"),
                        fused("affect", subject="departed", state="stale", now=990,
                              reference="old-evidence"),
                        fused("affect", subject="current"),
                    ]}}
        result = assemble_turn_context({}, snapshot, now=1001)
        self.assertEqual(result["perception"]["affect_cues"][0]["track_id"], "current")
        self.assertEqual(result["evidence"]["affect"]["state"], "available")
        self.assertEqual(result["evidence"]["affect"]["evidence_refs"], ["current-evidence"])

    def test_current_affect_must_have_same_source_subject_and_public_scope(self):
        for wrong in (fused("affect", subject="other"),
                      fused("affect", source="camera.environment", subject="current"),
                      fused("affect", subject="current", session_id="other-session")):
            with self.subTest(wrong=wrong):
                snapshot = {"people": [observation(track_id="current")],
                            "affect_cues": [observation(track_id="current", top_label="neutral")],
                            "fusion": {"ready": True, "items": [fused("presence"), wrong]}}
                result = assemble_turn_context({}, snapshot, now=1001)
                self.assertEqual(result["perception"]["affect_cues"], [])
                self.assertEqual(result["evidence"]["affect"]["state"], "unknown")
                self.assertEqual(result["evidence"]["affect"]["evidence_refs"], [])

    def test_current_stale_affect_is_not_replaced_by_other_fresh_track(self):
        snapshot = {"people": [observation(track_id="current")],
                    "affect_cues": [observation(track_id="current", top_label="neutral")],
                    "fusion": {"ready": True, "items": [
                        fused("presence"), fused("affect", subject="current", now=990,
                                                 state="stale", reference="current-stale"),
                        fused("affect", subject="other", reference="unrelated"),
                    ]}}
        result = assemble_turn_context({}, snapshot, now=1001)
        self.assertEqual(result["perception"]["affect_cues"], [])
        self.assertEqual(result["evidence"]["affect"]["state"], "stale")
        self.assertEqual(result["evidence"]["affect"]["evidence_refs"], ["current-stale"])

    def test_unused_live_cue_is_not_reported_as_present(self):
        snapshot = {"people": [observation(track_id="current")],
                    "affect_cues": [observation(now=990, track_id="current", top_label="neutral")],
                    "fusion": {"ready": True, "items": [
                        fused("presence"), fused("affect", subject="current"),
                    ]}}
        result = assemble_turn_context({}, snapshot, now=1001)
        self.assertEqual(result["perception"]["affect_cues"], [])
        self.assertEqual(result["evidence"]["affect"]["state"], "unknown")
        self.assertEqual(result["evidence"]["affect"]["reason"], "no_eligible_observation")

    def test_multiple_people_do_not_adopt_or_advertise_a_single_affect(self):
        snapshot = {"people": [observation(track_id="first"), observation(track_id="second")],
                    "affect_cues": [observation(track_id="first", top_label="neutral")],
                    "fusion": {"ready": True, "items": [
                        fused("presence"), fused("affect", subject="first"),
                    ]}}
        result = assemble_turn_context({}, snapshot, now=1001)
        self.assertEqual(len(result["perception"]["people"]), 2)
        self.assertEqual(result["perception"]["affect_cues"], [])
        self.assertEqual(result["evidence"]["affect"]["state"], "unknown")
        self.assertEqual(result["evidence"]["affect"]["evidence_refs"], [])

    def test_source_level_failure_and_expiry_still_filter_people(self):
        for state in ("unknown", "unavailable", "disabled", "observed_absent", "stale"):
            with self.subTest(state=state):
                snapshot = {"people": [observation(track_id="current")],
                            "fusion": {"ready": True, "items": [fused("presence", state=state)]}}
                result = assemble_turn_context({}, snapshot, now=1001)
                self.assertEqual(result["perception"]["people"], [])
                self.assertEqual(result["evidence"]["people"]["state"], state)
        snapshot["fusion"]["items"] = [fused("presence", now=990)]
        result = assemble_turn_context({}, snapshot, now=1001)
        self.assertEqual(result["perception"]["people"], [])
        self.assertEqual(result["evidence"]["people"]["state"], "stale")

    def test_fusion_does_not_bind_visible_face_to_voice_or_revive_closed_host(self):
        snapshot = {"people": [observation(track_id="current", identity=identity_cue("Synthetic", "current"))],
                    "fusion": {"ready": True, "items": [fused("presence")]}}
        identity = {"status": "known", "label": "Synthetic", "current_verified": True}
        result = assemble_turn_context({"input_type": "voice"}, snapshot, identity, now=1001)
        self.assertFalse(result["identity_verified"])
        self.assertIsNone(result["perception"]["people"][0]["identity"])
        snapshot["fusion"]["ready"] = False
        result = assemble_turn_context({}, snapshot, identity, now=1001)
        self.assertEqual(result["perception"]["people"], [])
        self.assertEqual(result["evidence"]["people"]["state"], "unavailable")

    def test_identity_has_its_own_freshness_source_and_track(self):
        valid = identity_cue("Synthetic", "current")
        for invalid in ({k: v for k, v in valid.items() if k != "observed_at"},
                        {k: v for k, v in valid.items() if k != "expires_at"},
                        {**valid, "expires_at": 1001}, {**valid, "observed_at": 1002},
                        {**valid, "source": "camera.environment"},
                        {**valid, "associated_track_id": "other"}):
            with self.subTest(identity=invalid):
                snapshot = {"people": [observation(track_id="current", identity=invalid)]}
                result = assemble_turn_context({}, snapshot, {
                    "status": "known", "label": "Synthetic", "current_verified": True,
                }, now=1001)
                self.assertEqual(len(result["perception"]["people"]), 1)
                self.assertFalse(result["identity_verified"])
                self.assertIsNone(result["perception"]["people"][0]["identity"])


if __name__ == "__main__":
    unittest.main()
