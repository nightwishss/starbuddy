#!/usr/bin/env python3

import threading
import unittest
from unittest.mock import Mock

from gateway.agent.router import LocalFirstRouter, NPUAdmission, ResourceBusy, RouteError
from governance import DataLabel, PrivacyContext


class ModelRouterTests(unittest.TestCase):
    def test_local_is_preferred_for_private_conversation(self):
        local, cloud = Mock(return_value="本地回答"), Mock(return_value="云回答")
        result = LocalFirstRouter(local, cloud).route(
            "你好", PrivacyContext(frozenset({DataLabel.CHILD_PRIVATE}), local_only=True)
        )
        self.assertEqual(result.backend, "local-agent")
        cloud.assert_not_called()

    def test_cloud_fallback_requires_consent(self):
        local = Mock(side_effect=RouteError("offline"))
        cloud = Mock(return_value="云回答")
        router = LocalFirstRouter(local, cloud)
        with self.assertRaisesRegex(RouteError, "cloud denied"):
            router.route("问题", PrivacyContext(frozenset({DataLabel.PERSONAL})))
        result = router.route(
            "问题", PrivacyContext(frozenset({DataLabel.PERSONAL}), guardian_cloud_consent=True)
        )
        self.assertTrue(result.escalated)

    def test_fallback_is_denied_after_side_effect(self):
        router = LocalFirstRouter(Mock(side_effect=RouteError("failed")), Mock(return_value="cloud"))
        with self.assertRaisesRegex(RouteError, "after side effect"):
            router.route(
                "操作", PrivacyContext(frozenset({DataLabel.PUBLIC}), guardian_cloud_consent=True),
                side_effects_executed=True,
            )

    def test_npu_admission_is_bounded(self):
        admission = NPUAdmission(slots=1)
        entered = threading.Event()
        release = threading.Event()
        thread = threading.Thread(target=lambda: self._hold(admission, entered, release))
        thread.start(); entered.wait(1)
        with self.assertRaises(ResourceBusy):
            with admission.acquire(timeout=0.01):
                pass
        release.set(); thread.join(1)

    @staticmethod
    def _hold(admission, entered, release):
        with admission.acquire():
            entered.set(); release.wait(1)


if __name__ == "__main__":
    unittest.main()
