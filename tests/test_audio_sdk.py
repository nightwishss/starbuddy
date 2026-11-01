#!/usr/bin/env python3
"""Application SDK tests for physical audio."""

import unittest
from unittest.mock import MagicMock

from sdk.audio import AudioClient, AudioError


class AudioSdkTests(unittest.TestCase):
    def test_uses_logical_roles_and_media_ids(self):
        runtime = MagicMock()
        runtime.call.side_effect = [
            {"ok": True, "result": {"media_id": "audio_test"}},
            {"ok": True, "result": {"media_id": "audio_turn"}},
            {"ok": True, "result": {"playback_id": "playback_test"}},
            {"ok": True, "result": {"mode": "half_duplex"}},
        ]
        client = AudioClient(runtime)
        client.capture_clip("session", duration_ms=2000, volume_percent=75)
        client.capture_turn("session", max_wait_ms=3000)
        client.start_playback("generated_test", "session", volume_percent=100)
        policy = client.interaction_status()
        self.assertEqual(runtime.call.call_args_list[0].args[:2], (
            "audio.capture_clip", {
                "media_owner": "session", "role": "audio.primary_input",
                "duration_ms": 2000, "sample_rate": 16000, "channels": 1,
                "volume_percent": 75,
            }))
        self.assertEqual(runtime.call.call_args_list[1].args[0], "audio.capture_turn")
        self.assertEqual(runtime.call.call_args_list[2].args[:2], (
            "audio.start_playback", {
                "media_id": "generated_test", "media_owner": "session",
                "role": "audio.primary_output", "volume_percent": 100,
            }))
        self.assertEqual(policy["mode"], "half_duplex")
        self.assertEqual(runtime.call.call_args_list[3].args[:2], (
            "audio.interaction_status", {},
        ))

    def test_structured_runtime_error(self):
        runtime = MagicMock()
        runtime.call.return_value = {
            "ok": False, "error": {"code": "HARDWARE_ERROR", "message": "missing"}}
        with self.assertRaises(AudioError) as raised:
            AudioClient(runtime).capture_clip("session")
        self.assertEqual(raised.exception.code, "HARDWARE_ERROR")


if __name__ == "__main__":
    unittest.main()
