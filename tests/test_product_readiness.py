#!/usr/bin/env python3
"""Deterministic readiness aggregation and cache tests."""

import unittest
from unittest.mock import Mock

from gateway.health import ProductReadiness


def runtime_ready():
    return {
        "ok": True, "speech_ready": True,
        "roles": {
            "audio.primary_input": "resolved", "audio.primary_output": "resolved",
            "camera.front": "resolved", "camera.environment": "resolved",
            "expression.matrix": "resolved", "display.primary": "missing",
        },
    }


class ProductReadinessTests(unittest.TestCase):
    def probes(self):
        return {
            "runtime": Mock(side_effect=runtime_ready),
            "media": Mock(return_value={"ok": True}),
            "conversation_session": Mock(return_value={"ok": True}),
            "local_llm": Mock(return_value={"ok": True}),
            "local_agent": Mock(return_value={"ok": True}),
            "vision_runner": Mock(return_value={"ok": True}),
            "display": Mock(return_value={"ok": True}),
            "telemetry": Mock(return_value={"ok": True}),
        }

    def test_all_product_features_are_independently_ready(self):
        result = ProductReadiness(probes=self.probes(), cache_seconds=0).collect()
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "ready")
        self.assertTrue(all(result["features"].values()))
        self.assertNotIn("error", result["components"]["runtime"])

    def test_optional_physical_hdmi_does_not_break_display_surface(self):
        probes = self.probes()
        result = ProductReadiness(probes=probes, cache_seconds=0).collect()
        self.assertTrue(result["features"]["display"])
        self.assertEqual(result["components"]["runtime"]["roles"]["display.primary"], "missing")

    def test_failed_feature_is_degraded_but_core_remains_available(self):
        probes = self.probes()
        probes["vision_runner"] = Mock(side_effect=OSError("private detail"))
        result = ProductReadiness(probes=probes, cache_seconds=0).collect()
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "degraded")
        self.assertFalse(result["features"]["vision"])
        self.assertEqual(result["components"]["vision_runner"]["detail"], "unavailable")

    def test_conversation_session_is_a_core_readiness_dependency(self):
        probes = self.probes()
        probes["conversation_session"] = Mock(return_value={"ok": False})
        result = ProductReadiness(probes=probes, cache_seconds=0).collect()
        self.assertFalse(result["features"]["conversation"])
        self.assertEqual(result["status"], "unavailable")

    def test_short_cache_prevents_probe_storm(self):
        probes = self.probes()
        readiness = ProductReadiness(probes=probes, cache_seconds=60)
        first = readiness.collect()
        first["components"]["runtime"]["roles"].clear()
        second = readiness.collect()
        self.assertTrue(second["components"]["runtime"]["roles"])
        for probe in probes.values():
            probe.assert_called_once()

    def test_runtime_probe_uses_application_sdk_contracts(self):
        runtime = Mock()
        runtime.call.side_effect = [
            {"ok": True, "result": {"healthy": True, "roles": {
                "audio.primary_input": {"status": "resolved"},
            }}},
            {"ok": True, "result": {"ready": True, "stt": True, "tts": True}},
        ]
        readiness = ProductReadiness(runtime=runtime, media=Mock(), telemetry=Mock())
        result = readiness._runtime()
        self.assertTrue(result["ok"])
        self.assertEqual(result["roles"]["audio.primary_input"], "resolved")
        self.assertEqual(runtime.call.call_args_list[0].args[:2], ("device.status", {}))
        self.assertEqual(runtime.call.call_args_list[0].kwargs, {"timeout_ms": 1_200})
        self.assertEqual(runtime.call.call_args_list[1].args[:2], ("speech.status", {}))


if __name__ == "__main__":
    unittest.main()
