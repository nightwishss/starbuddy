#!/usr/bin/env python3

import unittest
from unittest.mock import MagicMock, call

from sdk.speech import SpeechClient, SpeechError


class SpeechSdkTests(unittest.TestCase):
    def test_shared_media_is_mandatory_for_stt_and_tts(self):
        runtime = MagicMock()
        runtime.call.side_effect = [
            {"ok": True, "result": {"text": "你好"}},
            {"ok": True, "result": {"media_id": "generated_test"}},
        ]
        client = SpeechClient(runtime)
        self.assertEqual(client.transcribe("audio_test", "session:test")["text"], "你好")
        client.synthesize("你好", "session:test")
        self.assertEqual(runtime.call.call_args_list, [
            call("speech.transcribe", {
                "media_id": "audio_test", "media_owner": "session:test",
            }, timeout_ms=30_000),
            call("speech.synthesize", {
                "text": "你好", "media_owner": "session:test",
            }, timeout_ms=30_000),
        ])

    def test_owner_validation_and_runtime_error(self):
        with self.assertRaises(ValueError):
            SpeechClient(MagicMock()).transcribe("audio_test", "")
        runtime = MagicMock()
        runtime.call.return_value = {"ok": False, "error": {"code": "HARDWARE_ERROR", "message": "failed"}}
        with self.assertRaises(SpeechError) as raised:
            SpeechClient(runtime).status()
        self.assertEqual(raised.exception.domain, "speech")


if __name__ == "__main__":
    unittest.main()
