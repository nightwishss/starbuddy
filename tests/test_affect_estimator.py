#!/usr/bin/env python3

import unittest

from perception import AffectEstimator, AffectFeatures


class AffectEstimatorTests(unittest.TestCase):
    def setUp(self): self.estimator = AffectEstimator()

    def test_explicit_negative_text_selects_gentle_candidate(self):
        result = self.estimator.estimate(AffectFeatures(text="我今天有点难过"))
        self.assertEqual(result.label, "gentle_support")
        self.assertLess(result.valence, 0)
        self.assertIn("explicit_negative_text", result.evidence)

    def test_multiple_features_raise_bounded_confidence(self):
        result = self.estimator.estimate(AffectFeatures(text="我很紧张", rms=0.9, speech_rate=0.9))
        self.assertGreater(result.arousal, 0)
        self.assertGreater(result.confidence, 0.5)
        self.assertLessEqual(result.confidence, 1)

    def test_neutral_without_evidence_has_zero_confidence(self):
        result = self.estimator.estimate(AffectFeatures(text="今天天气怎么样"))
        self.assertEqual(result.label, "neutral")
        self.assertEqual(result.confidence, 0)

    def test_invalid_features_are_rejected(self):
        with self.assertRaises(ValueError): AffectFeatures(rms=2)


if __name__ == "__main__": unittest.main()
