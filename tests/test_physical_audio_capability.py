#!/usr/bin/env python3
"""Contract and boundary tests for physical PipeWire audio capabilities."""

import tempfile
import unittest
import wave
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from capability.audio import physical
from runtime.errors import ResourceNotFoundError, SchemaValidationError


def write_wav(path: Path, frames=1600, value=1000):
    with wave.open(str(path), "wb") as stream:
        stream.setframerate(16000)
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.writeframes(int(value).to_bytes(2, "little", signed=True) * frames)


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.signals = []

    def poll(self):
        return self.returncode

    def send_signal(self, value):
        self.signals.append(value)
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


class PhysicalAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        physical._PLAYBACKS.clear()

    def tearDown(self):
        physical._PLAYBACKS.clear()
        self.temp.cleanup()

    @patch.object(physical, "_media_client")
    @patch.object(physical, "_run_capture")
    @patch.object(physical, "_resolve_role")
    def test_capture_uses_logical_role_and_commits_media(self, resolve, run_capture, media_client):
        resolve.return_value = (
            {"settings": {"volume_percent": 75}},
            {"endpoint": "pipewire.input", "device_id": "audio_input:usb"},
        )
        run_capture.side_effect = lambda target, output, *args: write_wav(output)
        media_client.return_value.create_from_path.return_value = {
            "media_id": "audio_ABCDEFGHIJKLMNOPQRSTUVWXYZ", "kind": "audio",
            "mime_type": "audio/wav", "size": 3244, "created_at": 1.0,
            "expires_at": 601.0,
        }

        result = physical.capture_clip({"media_owner": "phone-session", "duration_ms": 1000})

        self.assertEqual(result["role"], "audio.primary_input")
        self.assertEqual(result["volume_percent"], 75)
        self.assertFalse(result["signal"]["silent"])
        resolve.assert_called_once_with("audio.primary_input", "audio_input", "audio.capture_clip")
        self.assertEqual(media_client.return_value.create_from_path.call_args.args[:2],
                         ("phone-session", "audio/wav"))
        self.assertFalse(media_client.return_value.create_from_path.call_args.args[2].exists())

    @patch.object(physical.subprocess, "Popen")
    @patch.object(physical, "_shared_wav")
    @patch.object(physical, "_resolve_role")
    def test_playback_is_status_queryable_and_interruptible(self, resolve, shared, popen):
        wav = self.root / "reply.wav"
        write_wav(wav)
        resolve.return_value = (
            {"settings": {"volume_percent": 100}},
            {"endpoint": "pipewire.output", "device_id": "audio_output:usb"},
        )
        shared.return_value = (
            wav, {"media_id": "generated_ABCDEFGHIJKLMNOPQRSTUVWXYZ", "mime_type": "audio/wav"}
        )
        process = FakeProcess()
        popen.return_value = process

        started = physical.start_playback({
            "media_id": "generated_ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "media_owner": "phone-session", "volume_percent": 80,
        })
        self.assertEqual(started["state"], "playing")
        self.assertEqual(physical.playback_status(
            {"playback_id": started["playback_id"]})["state"], "playing")
        stopped = physical.stop_playback({"playback_id": started["playback_id"]})
        self.assertEqual(stopped["state"], "stopped")
        self.assertTrue(process.signals)
        with self.assertRaises(ResourceNotFoundError):
            physical.playback_status({"playback_id": started["playback_id"]})

    def test_owner_and_direction_are_validated(self):
        with self.assertRaises(SchemaValidationError):
            physical._validate_owner("bad owner", "audio.capture_clip")
        with patch.object(physical.HardwareProfileManager, "resolve") as resolve:
            resolve.return_value = {
                "kind": "audio_output", "settings": {},
                "device": {"endpoint": "output"},
            }
            with self.assertRaises(SchemaValidationError):
                physical._resolve_role("audio.primary_output", "audio_input", "audio.capture_clip")

    @patch.object(physical.HardwareProfileManager, "resolve_all")
    @patch.object(physical.subprocess, "run")
    def test_interaction_status_reports_available_graph_without_claiming_product_aec(
            self, run, resolve_all):
        run.return_value = subprocess.CompletedProcess(
            [], 0, "7\tmodule-echo-cancel\taec_method=webrtc\n", ""
        )
        resolve_all.return_value = {"roles": {
            "audio.echo_cancelled_input": {"status": "resolved"},
            "audio.echo_reference_output": {"status": "resolved"},
        }}

        result = physical.interaction_status({})

        self.assertEqual(result["mode"], "half_duplex")
        self.assertEqual(result["trigger"], "explicit")
        self.assertEqual(result["echo_guard_ms"], 180)
        self.assertTrue(result["aec"]["module_loaded"])
        self.assertTrue(result["aec"]["available"])
        self.assertFalse(result["aec"]["enabled"])
        self.assertFalse(result["wake_word"]["background_capture"])

    @patch.object(physical.subprocess, "Popen")
    def test_timed_sigint_accepts_pipewire_exit_one(self, popen):
        process = FakeProcess()
        process.wait = MagicMock(side_effect=[subprocess.TimeoutExpired("pw-record", 1), 1])
        popen.return_value = process
        physical._run_capture("input", self.root / "capture.wav", 1, 16000, 1, 75)
        self.assertTrue(process.signals)

    @patch.object(physical, "_media_client")
    @patch.object(physical, "_capture_turn_pcm")
    @patch.object(physical, "_resolve_role")
    def test_natural_turn_commits_only_detected_speech(self, resolve, capture, media_client):
        resolve.return_value = (
            {"settings": {"volume_percent": 75}},
            {"endpoint": "pipewire.input", "device_id": "audio_input:usb"},
        )
        capture.return_value = (
            int(2000).to_bytes(2, "little", signed=True) * 1600,
            {"detected": True, "end_reason": "end_silence", "noise_floor_dbfs": -60.0},
        )
        media_client.return_value.create_from_path.return_value = {
            "media_id": "audio_ABCDEFGHIJKLMNOPQRSTUVWXYZ", "kind": "audio",
            "mime_type": "audio/wav", "size": 3244, "created_at": 1.0,
            "expires_at": 601.0,
        }
        result = physical.capture_turn({"media_owner": "phone-session"})
        self.assertTrue(result["detected"])
        self.assertEqual(result["turn"]["end_reason"], "end_silence")
        self.assertEqual(result["signal"]["sample_rate"], 16000)

        capture.return_value = (None, {"detected": False, "end_reason": "no_speech"})
        result = physical.capture_turn({"media_owner": "phone-session"})
        self.assertFalse(result["detected"])
        self.assertNotIn("media_id", result)


if __name__ == "__main__":
    unittest.main()
