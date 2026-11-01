#!/usr/bin/env python3

import unittest

from governance import ChildSafetyPolicy, SafetyLevel


class ChildSafetyPolicyTests(unittest.TestCase):
    def setUp(self): self.policy = ChildSafetyPolicy()

    def test_immediate_danger_bypasses_model(self):
        decision = self.policy.evaluate("我不想活了")
        self.assertEqual(decision.level, SafetyLevel.IMMEDIATE)
        self.assertFalse(decision.allow_model)
        self.assertTrue(decision.require_trusted_adult)
        self.assertIn("可信任的大人", decision.response)

    def test_distress_allows_bounded_support(self):
        decision = self.policy.evaluate("我今天有点难过")
        self.assertEqual(decision.level, SafetyLevel.DISTRESS)
        self.assertTrue(decision.allow_model)

    def test_normal_text_is_not_escalated(self):
        self.assertEqual(self.policy.evaluate("我们画一颗星星").level, SafetyLevel.NORMAL)


if __name__ == "__main__": unittest.main()
