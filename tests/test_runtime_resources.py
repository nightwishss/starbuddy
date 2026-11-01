#!/usr/bin/env python3
"""Resource ownership lifecycle tests."""

import unittest

from runtime.errors import PermissionDeniedError
from runtime.resources import ResourceTracker


class ResourceTrackerTests(unittest.TestCase):
    def test_terminal_playback_status_releases_runtime_handle(self):
        tracker = ResourceTracker()
        tracker.record_result(
            1000, "audio.start_playback", {}, {"playback_id": "playback_test"},
        )
        tracker.authorize(1000, {"playback_id": "playback_test"})
        tracker.record_result(
            1000, "audio.playback_status", {"playback_id": "playback_test"},
            {"playback_id": "playback_test", "state": "completed"},
        )
        with self.assertRaises(PermissionDeniedError):
            tracker.authorize(1000, {"playback_id": "playback_test"})
        self.assertEqual(tracker.records(), [])

    def test_playing_status_retains_runtime_handle(self):
        tracker = ResourceTracker()
        tracker.record_result(
            1000, "audio.start_playback", {}, {"playback_id": "playback_test"},
        )
        tracker.record_result(
            1000, "audio.playback_status", {"playback_id": "playback_test"},
            {"playback_id": "playback_test", "state": "playing"},
        )
        tracker.authorize(1000, {"playback_id": "playback_test"})


if __name__ == "__main__":
    unittest.main()
