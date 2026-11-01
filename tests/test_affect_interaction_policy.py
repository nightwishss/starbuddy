import unittest

from interaction_policy import AffectInteractionPolicy


def snapshot(label="neutral", confidence=.9, *, expires_at=1010, track_id="face_front_1"):
    return {
        "people": [{"source": "camera.front", "track_id": "face_front_1"}],
        "affect_cues": [{"source": "camera.front", "track_id": track_id,
                         "top_label": label, "confidence": confidence,
                         "expires_at": expires_at}],
    }


class AffectInteractionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = AffectInteractionPolicy(clock=lambda: 1000)

    def test_explicit_request_preserves_uncertainty(self):
        decision = self.policy.decide("你觉得我现在心情怎么样？", snapshot("sad_like"))
        self.assertIn("不一定看得准", decision.direct_answer)
        self.assertIn("你自己感觉", decision.direct_answer)
        self.assertEqual(decision.visual_style, "gentle")

    def test_low_confidence_stale_or_ambiguous_face_does_not_claim_emotion(self):
        for current in (
            snapshot(confidence=.5), snapshot(expires_at=999),
            snapshot(track_id="another"), {**snapshot(), "people": []},
            {**snapshot(), "people": snapshot()["people"] * 2},
        ):
            answer = self.policy.decide("我的状态怎么样？", current).direct_answer
            self.assertIn("没有可靠的表情线索", answer)

    def test_normal_chat_has_only_tone_hint_and_never_names_cue(self):
        decision = self.policy.decide("给我讲个故事", snapshot("sad_like"))
        self.assertIsNone(decision.direct_answer)
        self.assertIn("语气温和", decision.tone_hint)
        self.assertIn("不要提到", decision.tone_hint)
        self.assertIsNone(self.policy.decide("我现在心情不好", snapshot("sad_like")).direct_answer)
        self.assertIsNone(self.policy.decide("你的状态怎么样？", snapshot("sad_like")).direct_answer)


if __name__ == "__main__":
    unittest.main()
