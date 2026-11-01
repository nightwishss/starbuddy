#!/usr/bin/env python3
"""Tests for the model-independent ergonomic reminder policy."""

import unittest

from perception import PostureSignal, PostureTemporalPolicy


class PostureTemporalPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = PostureTemporalPolicy(
            confirmation_seconds=20, reminder_cooldown_seconds=180,
            recovery_seconds=5, minimum_confidence=0.6,
        )

    def signal(self, at, *, visible=True, confidence=0.9, reasons=()):
        return PostureSignal(at, visible, confidence, tuple(reasons))

    def test_poor_visibility_is_unknown_and_never_a_bad_posture(self):
        decision = self.policy.update(self.signal(0, visible=False, confidence=0.1))
        self.assertEqual(decision.state, "unknown")
        self.assertFalse(decision.remind)

    def test_continuous_deviation_confirms_once_then_obeys_cooldown(self):
        first = self.policy.update(self.signal(0, reasons=("head_forward",)))
        almost = self.policy.update(self.signal(19.9, reasons=("head_forward",)))
        confirmed = self.policy.update(self.signal(20, reasons=("head_forward",)))
        suppressed = self.policy.update(self.signal(30, reasons=("head_forward",)))
        repeated = self.policy.update(self.signal(200, reasons=("head_forward",)))
        self.assertEqual(first.state, "suspect")
        self.assertEqual(almost.state, "suspect")
        self.assertEqual(confirmed.state, "unhealthy")
        self.assertTrue(confirmed.remind)
        self.assertFalse(suppressed.remind)
        self.assertTrue(repeated.remind)

    def test_recovery_must_be_sustained_before_reset(self):
        self.policy.update(self.signal(0, reasons=("torso_lean",)))
        self.policy.update(self.signal(20, reasons=("torso_lean",)))
        recovering = self.policy.update(self.signal(21))
        healthy = self.policy.update(self.signal(26))
        self.assertEqual(recovering.state, "suspect")
        self.assertFalse(recovering.remind)
        self.assertEqual(healthy.state, "healthy")

    def test_low_confidence_cannot_trigger_reminder(self):
        self.policy.update(self.signal(0, confidence=0.4, reasons=("too_close",)))
        decision = self.policy.update(
            self.signal(60, confidence=0.4, reasons=("too_close",))
        )
        self.assertEqual(decision.state, "unknown")
        self.assertFalse(decision.remind)

    def test_inputs_are_bounded(self):
        with self.assertRaises(ValueError):
            PostureSignal(0, True, 1.2)
        with self.assertRaises(ValueError):
            PostureSignal(0, True, 0.8, ("diagnosis",))
        self.policy.update(self.signal(5))
        with self.assertRaises(ValueError):
            self.policy.update(self.signal(4))


if __name__ == "__main__":
    unittest.main()
