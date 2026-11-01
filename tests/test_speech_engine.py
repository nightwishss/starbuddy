#!/usr/bin/env python3
"""Tests for board-local speech command adapters."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gateway.speech import SpeechEngine, SpeechError, prepare_tts_text


class SpeechEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.paths = [root / name for name in ("whisper", "stt.bin", "python", "tts.onnx", "ffmpeg")]
        for path in self.paths:
            path.write_bytes(b"test")
        self.engine = SpeechEngine(
            *self.paths[:4], ffmpeg=self.paths[4], timeout=5, threads=4,
            persistent_tts=False,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_ready_requires_all_runtime_files(self):
        self.assertTrue(self.engine.ready())
        self.paths[1].unlink()
        self.assertFalse(self.engine.ready())

    @patch("gateway.speech.engine.subprocess.run")
    def test_transcribe_uses_safe_argument_vectors(self, run):
        def complete(command, **kwargs):
            if "-otxt" in command:
                Path(command[command.index("-of") + 1] + ".txt").write_text("你好板卡\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")

        run.side_effect = complete
        source = Path(self.temp.name) / "phone.webm"
        source.write_bytes(b"webm")
        self.assertEqual(self.engine.transcribe(source, Path(self.temp.name) / "work"), "你好板卡")
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertIsInstance(call.args[0], list)
            self.assertNotIn("shell", call.kwargs)

    @patch("gateway.speech.engine.subprocess.run")
    def test_synthesis_bounds_spoken_text(self, run):
        def complete(command, **kwargs):
            Path(command[command.index("-f") + 1]).write_bytes(b"RIFFxxxxWAVEdata")
            return subprocess.CompletedProcess(command, 0, "", "")

        run.side_effect = complete
        output = Path(self.temp.name) / "answer.wav"
        self.engine.synthesize("字" * 3000, output)
        command = run.call_args.args[0]
        self.assertLessEqual(len(command[-1]), 2020)
        self.assertEqual(command[command.index("--sentence-silence") + 1], "0.14")
        self.assertEqual(command[command.index("--length-scale") + 1], "0.92")
        self.assertTrue(output.exists())

    def test_display_markup_is_normalized_for_speech(self):
        self.assertEqual(
            prepare_tts_text("## 建议\n- 先深呼吸；\n- 再慢慢说。。"),
            "建议，先深呼吸，再慢慢说。",
        )

    def test_bilingual_boundaries_preserve_english_name(self):
        self.assertEqual(
            prepare_tts_text("你好April，welcome回来"),
            "你好 April，welcome 回来",
        )

    def test_matcha_is_preferred_when_its_assets_are_complete(self):
        root = Path(self.temp.name)
        model = root / "matcha"
        model.mkdir()
        for name in ("model-steps-3.onnx", "lexicon.txt", "tokens.txt"):
            (model / name).write_bytes(b"test")
        vocoder = root / "vocoder.onnx"
        vocoder.write_bytes(b"test")
        engine = SpeechEngine(
            *self.paths[:4], ffmpeg=self.paths[4], tts_backend="matcha",
            matcha_python=self.paths[2], matcha_model_dir=model,
            matcha_vocoder=vocoder,
        )
        self.assertTrue(engine.tts_ready())
        self.assertEqual(engine.active_tts_backend(), "matcha")

    def test_matcha_configuration_falls_back_to_piper_if_assets_are_missing(self):
        engine = SpeechEngine(
            *self.paths[:4], ffmpeg=self.paths[4], tts_backend="matcha",
        )
        self.assertTrue(engine.tts_ready())
        self.assertEqual(engine.active_tts_backend(), "piper")

    def test_mixed_chinese_english_selects_bilingual_model(self):
        root = Path(self.temp.name)
        model = root / "melo"
        model.mkdir()
        for name in ("model.onnx", "lexicon.txt", "tokens.txt"):
            (model / name).write_bytes(b"test")
        engine = SpeechEngine(
            *self.paths[:4], ffmpeg=self.paths[4],
            melo_python=self.paths[2], melo_model_dir=model,
        )
        self.assertEqual(engine.tts_backend_for("你好，welcome back"), "melo-zh-en")
        self.assertEqual(engine.tts_backend_for("欢迎回来"), "piper")

    def test_unified_melo_backend_uses_one_voice_for_all_supported_text(self):
        root = Path(self.temp.name)
        model = root / "melo-unified"
        model.mkdir()
        for name in ("model.onnx", "lexicon.txt", "tokens.txt"):
            (model / name).write_bytes(b"test")
        engine = SpeechEngine(
            *self.paths[:4], ffmpeg=self.paths[4], tts_backend="melo",
            melo_python=self.paths[2], melo_model_dir=model,
        )
        self.assertTrue(engine.tts_ready())
        self.assertEqual(engine.active_tts_backend(), "melo-zh-en")
        self.assertEqual(engine.tts_backend_for("欢迎回来"), "melo-zh-en")
        self.assertEqual(engine.tts_backend_for("April，欢迎回来"), "melo-zh-en")
        self.assertEqual(engine.tts_backend_for("Welcome back"), "melo-zh-en")

    def test_unified_melo_backend_fails_closed_when_assets_are_missing(self):
        engine = SpeechEngine(
            *self.paths[:4], ffmpeg=self.paths[4], tts_backend="melo",
        )
        self.assertFalse(engine.tts_ready())
        self.assertEqual(engine.active_tts_backend(), "unavailable")
        with self.assertRaisesRegex(SpeechError, "统一语音模型尚未就绪"):
            engine.tts_backend_for("你好")

    def test_english_does_not_silently_fall_back_to_chinese_voice(self):
        with self.assertRaisesRegex(SpeechError, "英文语音模型尚未就绪"):
            self.engine.tts_backend_for("Hello StarBuddy")

    @patch("gateway.speech.engine.subprocess.run")
    def test_runtime_failure_is_not_leaked(self, run):
        run.return_value = subprocess.CompletedProcess([], 1, "", "private model path")
        with self.assertRaisesRegex(SpeechError, "语音合成失败"):
            self.engine.synthesize("你好", Path(self.temp.name) / "failed.wav")


if __name__ == "__main__":
    unittest.main()
