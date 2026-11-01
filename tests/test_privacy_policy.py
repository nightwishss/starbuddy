#!/usr/bin/env python3

import unittest

from governance import DataLabel, Destination, PrivacyContext, PrivacyPolicy


class PrivacyPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = PrivacyPolicy()

    def test_local_is_always_allowed(self):
        context = PrivacyContext(frozenset({DataLabel.CHILD_PRIVATE}), local_only=True)
        self.assertTrue(self.policy.decide(context, Destination.LOCAL).allowed)

    def test_child_private_and_sensitive_never_reach_cloud(self):
        for label in (DataLabel.CHILD_PRIVATE, DataLabel.SENSITIVE):
            context = PrivacyContext(frozenset({label}), guardian_cloud_consent=True)
            self.assertFalse(self.policy.decide(context, Destination.CLOUD_TEXT).allowed)

    def test_raw_biometric_media_is_denied_even_with_consent(self):
        context = PrivacyContext(frozenset({DataLabel.BIOMETRIC}), guardian_cloud_consent=True)
        self.assertFalse(self.policy.decide(context, Destination.CLOUD_MEDIA).allowed)
        self.assertTrue(self.policy.decide(context, Destination.CLOUD_TEXT).allowed)

    def test_cloud_text_requires_consent(self):
        context = PrivacyContext(frozenset({DataLabel.PERSONAL}))
        self.assertFalse(self.policy.cloud_text_allowed(context))
        consented = PrivacyContext(frozenset({DataLabel.PERSONAL}), guardian_cloud_consent=True)
        self.assertTrue(self.policy.cloud_text_allowed(consented))


if __name__ == "__main__":
    unittest.main()
