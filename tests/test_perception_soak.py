#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock

from tools import perception_soak


class PerceptionSoakTests(unittest.TestCase):
    def test_success_requires_no_media_growth(self):
        perception = MagicMock()
        perception.observe.return_value = {
            "stable_summary": {"counts": {"person": 1}},
            "change": {"changed": False, "provisional": False},
        }
        runtime = MagicMock()
        runtime.call.return_value = {"ok": True, "result": {}}
        media = MagicMock()
        media.call.side_effect = [{"count": 0}, {"count": 0}]

        result = perception_soak.run(
            samples=3, interval_seconds=0, perception=perception,
            runtime=runtime, media=media, clock=lambda: 1.0,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["samples_succeeded"], 3)
        self.assertEqual(result["stable_label_occurrences"], {"person": 3})
        self.assertTrue(result["media_clean"])

    def test_three_consecutive_failures_stop_early(self):
        perception = MagicMock()
        perception.observe.side_effect = RuntimeError("camera down")
        media = MagicMock()
        media.call.side_effect = [{"count": 0}, {"count": 0}]
        result = perception_soak.run(
            samples=10, interval_seconds=0, perception=perception,
            runtime=MagicMock(), media=media, clock=lambda: 1.0,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["failures"], 3)
        self.assertEqual(result["samples_attempted"], 3)

    def test_bounds_are_enforced_before_clients_are_created(self):
        with self.assertRaises(perception_soak.SoakError):
            perception_soak.run(samples=1, interval_seconds=3)
        with self.assertRaises(perception_soak.SoakError):
            perception_soak.run(samples=2, interval_seconds=61)


if __name__ == "__main__":
    unittest.main()
