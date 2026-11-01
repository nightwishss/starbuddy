#!/usr/bin/env python3

import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from tools import audio_aec_probe as probe


def write_segmented_wav(path: Path, noise: int, far: int, duration=2.0):
    rate = probe.SAMPLE_RATE
    values = []
    for index in range(int(rate * (duration + probe.CAPTURE_LEAD_SECONDS + 0.5))):
        value = noise if index < int(rate * probe.CAPTURE_LEAD_SECONDS) else far
        values.append(value)
    with wave.open(str(path), "wb") as stream:
        stream.setframerate(rate)
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.writeframes(b"".join(value.to_bytes(2, "little", signed=True) for value in values))


class AudioAecProbeTests(unittest.TestCase):
    def test_metrics_separate_noise_and_far_end_windows(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.wav"
            write_segmented_wav(path, 100, 2000)
            metrics = probe._metrics(path, 2.0)
        self.assertGreater(metrics["far_end_over_noise_db"], 20)
        self.assertEqual(metrics["sample_rate"], 48000)

    def test_bounds_reject_uncontrolled_volume_and_duration(self):
        with self.assertRaises(probe.ProbeError):
            probe.probe(execute=False, duration_seconds=6, volume_percent=20)
        with self.assertRaises(probe.ProbeError):
            probe.probe(execute=False, duration_seconds=3, volume_percent=41)

    @patch.object(probe, "_active_streams", return_value={"capture": 0, "playback": 0})
    @patch.object(probe, "_logical_audio")
    def test_default_is_non_mutating_inventory(self, logical, _streams):
        logical.return_value = (
            {"device_id": "audio_input:test", "endpoint": "input"},
            {"device_id": "audio_output:test", "endpoint": "output"},
        )
        result = probe.probe(execute=False, duration_seconds=3, volume_percent=20)
        self.assertEqual(result["status"], "dry_run")
        self.assertFalse(result["execute"])
        self.assertFalse(result["artifacts_retained"])

    @patch.object(probe.time, "sleep")
    @patch.object(probe, "_unload_aec")
    @patch.object(probe, "_load_aec", return_value="7")
    @patch.object(probe, "_capture_playback")
    @patch.object(probe, "_active_streams", return_value={"capture": 0, "playback": 0})
    @patch.object(probe, "_logical_audio")
    def test_execute_compares_captures_and_unloads_module(
            self, logical, _streams, capture, _load, unload, _sleep):
        logical.return_value = (
            {"device_id": "audio_input:test", "endpoint": "input"},
            {"device_id": "audio_output:test", "endpoint": "output"},
        )

        def create_capture(_source, _sink, _stimulus, output, duration, _volume):
            write_segmented_wav(output, 100, 2000 if output.name == "baseline.wav" else 500,
                                duration=duration)
        capture.side_effect = create_capture
        with tempfile.TemporaryDirectory() as runtime_dir, patch.dict(
                "os.environ", {"XDG_RUNTIME_DIR": runtime_dir}):
            result = probe.probe(
                execute=True, duration_seconds=2, volume_percent=20
            )

        self.assertTrue(result["preliminary_far_end_pass"])
        self.assertFalse(result["near_end_speech_tested"])
        self.assertFalse(result["product_aec_enabled"])
        unload.assert_called_once_with("7")

    @patch.object(probe.time, "sleep")
    @patch.object(probe, "_unload_aec")
    @patch.object(probe, "_load_aec", return_value="8")
    @patch.object(probe, "_capture_playback")
    @patch.object(probe, "_active_streams", return_value={"capture": 0, "playback": 0})
    @patch.object(probe, "_logical_audio")
    def test_failure_after_module_load_still_unloads(
            self, logical, _streams, capture, _load, unload, _sleep):
        logical.return_value = (
            {"device_id": "audio_input:test", "endpoint": "input"},
            {"device_id": "audio_output:test", "endpoint": "output"},
        )

        def fail_second(_source, _sink, _stimulus, output, duration, _volume):
            if output.name == "baseline.wav":
                write_segmented_wav(output, 100, 2000, duration=duration)
                return
            raise probe.ProbeError("aec capture failed")
        capture.side_effect = fail_second
        with tempfile.TemporaryDirectory() as runtime_dir, patch.dict(
                "os.environ", {"XDG_RUNTIME_DIR": runtime_dir}):
            with self.assertRaisesRegex(probe.ProbeError, "aec capture"):
                probe.probe(execute=True, duration_seconds=2, volume_percent=20)

        unload.assert_called_once_with("8")


if __name__ == "__main__":
    unittest.main()
