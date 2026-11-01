#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from tools import audio_double_talk_probe as probe


class AudioDoubleTalkProbeTests(unittest.TestCase):
    def test_similarity_and_gate_are_bounded_and_explainable(self):
        self.assertEqual(probe._similarity(probe.REFERENCE_TEXT), 1.0)
        self.assertLess(probe._similarity("完全不同"), 0.3)
        passed = probe._decision(0.9, 0.85, 0.7)
        failed = probe._decision(0.9, 0.5, 0.4)
        self.assertTrue(passed["preliminary_double_talk_pass"])
        self.assertFalse(failed["preliminary_double_talk_pass"])
        self.assertFalse(passed["product_aec_enabled"])

    def test_transcribe_always_releases_temporary_media(self):
        media = MagicMock()
        media.create_from_path.return_value = {"media_id": "audio_ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
        runtime = MagicMock()
        runtime.call.return_value = {"ok": False, "error": {"message": "failed"}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "speech.wav"
            path.write_bytes(b"RIFFtemporary")
            with self.assertRaises(probe.DoubleTalkError):
                probe._transcribe(path, media=media, runtime=runtime)
        media.call.assert_called_once_with(
            "release", owner=probe.OWNER, media_id="audio_ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )

    def test_duration_and_volume_are_bounded_before_hardware_access(self):
        with self.assertRaises(probe.DoubleTalkError):
            probe.run(duration=2, volume_percent=15)
        with self.assertRaises(probe.DoubleTalkError):
            probe.run(duration=5, volume_percent=31)


if __name__ == "__main__":
    unittest.main()
