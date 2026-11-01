#!/usr/bin/env python3
"""Security-boundary tests for Runtime speech capabilities."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from capability.speech import impl
from runtime.errors import ResourceNotFoundError, SchemaValidationError


class FakeEngine:
    whisper = Path("/test/whisper")
    whisper_model = Path("/test/whisper-model")
    piper_python = Path("/test/piper")
    piper_model = Path("/test/piper-model")

    def ready(self):
        return True

    def tts_ready(self):
        return True

    def bilingual_tts_ready(self):
        return True

    def active_tts_backend(self):
        return "matcha"

    def transcribe(self, source, work_dir):
        return "板端识别结果"

    def synthesize(self, text, output):
        output.write_bytes(b"RIFFxxxxWAVEdata")


class SpeechCapabilityTests(unittest.TestCase):
    def test_status_reports_model_files_without_running_inference(self):
        with patch.object(impl._ENGINE, "ready", return_value=True), \
             patch.object(Path, "is_file", return_value=True):
            result = impl.status({})
        self.assertEqual(result, {
            "ready": True, "stt": True, "tts": True,
            "tts_languages": ["zh-CN", "en-US"],
            "tts_bilingual": True,
            "tts_backends": {"default": "matcha", "mixed_zh_en": "melo-zh-en"},
        })

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.patches = [
            patch.object(impl, "_MEDIA_ROOT", self.root),
            patch.object(impl, "_ENGINE", FakeEngine()),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_transcribe_resolves_only_random_media_ids(self):
        media_id = "audio_ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        (self.root / f"{media_id}.webm").write_bytes(b"webm")
        self.assertEqual(impl.transcribe({"media_id": media_id})["text"], "板端识别结果")
        with self.assertRaises(SchemaValidationError):
            impl.transcribe({"media_id": "../../etc/passwd"})
        with self.assertRaises(ResourceNotFoundError):
            impl.transcribe({"media_id": "audio_ZYXWVUTSRQPONMLKJIHGFEDCBA"})

    def test_synthesize_returns_opaque_token_and_mode_0600_file(self):
        result = impl.synthesize({"text": "你好"})
        output = self.root / f"runtime-tts-{result['audio_token']}.wav"
        self.assertEqual(result["mime_type"], "audio/wav")
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)

    def test_synthesize_rejects_unbounded_text(self):
        with self.assertRaises(SchemaValidationError):
            impl.synthesize({"text": "x" * 2001})

    @patch.object(impl, "_media_client")
    def test_shared_media_owner_is_used_for_stt_and_tts(self, media_client):
        source = self.root / "shared.wav"
        source.write_bytes(b"RIFFxxxxWAVEdata")
        media_client.return_value.call.return_value = {"path": str(source)}
        media_client.return_value.create_from_path.return_value = {
            "media_id": "generated_ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "kind": "generated", "mime_type": "audio/wav", "size": 16,
            "created_at": 1.0, "expires_at": 601.0,
        }
        media_id = "audio_ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.assertEqual(
            impl.transcribe({"media_id": media_id, "media_owner": "phone-session"})["text"],
            "板端识别结果",
        )
        result = impl.synthesize({"text": "你好", "media_owner": "phone-session"})
        self.assertTrue(result["media_id"].startswith("generated_"))
        media_client.return_value.call.assert_called_with(
            "resolve", owner="phone-session", media_id=media_id)
        self.assertEqual(media_client.return_value.create_from_path.call_args.args[:2],
                         ("phone-session", "audio/wav"))


if __name__ == "__main__":
    unittest.main()
