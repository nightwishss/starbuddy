#!/usr/bin/env python3

import unittest

from observability import TraceStore


class TraceStoreTests(unittest.TestCase):
    def test_trace_records_only_stage_metadata(self):
        now = [0.0]
        store = TraceStore(clock=lambda: now[0])
        store.start("trace", "voice")
        with store.stage("trace", "stt"):
            now[0] = 0.2
        with store.stage("trace", "agent"):
            now[0] = 0.5
        store.set_backend("trace", "local-qwen")
        result = store.finish("trace", True)
        self.assertEqual(result["elapsed_ms"], 500.0)
        self.assertEqual([item["name"] for item in result["stages"]], ["stt", "agent"])
        self.assertTrue(result["budget"]["met"])
        self.assertEqual(result["backend"], "local-qwen")
        self.assertNotIn("text", str(result).lower())

    def test_failed_stage_and_ttl_are_visible(self):
        now = [0.0]
        store = TraceStore(ttl_seconds=2, clock=lambda: now[0])
        store.start("trace", "chat")
        with self.assertRaises(RuntimeError):
            with store.stage("trace", "agent"):
                raise RuntimeError("secret")
        self.assertFalse(store.get("trace")["stages"][0]["ok"])
        now[0] = 3
        self.assertIsNone(store.get("trace"))

    def test_budget_marks_stage_and_total_violations(self):
        now = [0.0]
        store = TraceStore(clock=lambda: now[0], budgets={
            "voice": {"total_ms": 1000, "stages": {"stt": 300}},
        })
        store.start("slow", "voice")
        with store.stage("slow", "stt"):
            now[0] = 0.4
        now[0] = 1.2
        result = store.finish("slow", True)

        self.assertFalse(result["budget"]["met"])
        self.assertEqual(
            [item["name"] for item in result["budget"]["violations"]],
            ["total", "stt"],
        )

    def test_summary_is_aggregate_only_with_p50_and_p95(self):
        now = [0.0]
        store = TraceStore(clock=lambda: now[0], budgets={
            "chat": {"total_ms": 250, "stages": {"decision": 200}},
        })
        for index, elapsed in enumerate((0.1, 0.2, 0.4)):
            now[0] = index
            store.start(f"private-{index}", "chat")
            with store.stage(f"private-{index}", "decision"):
                now[0] = index + elapsed
            store.set_backend(f"private-{index}", "local-qwen")
            store.finish(f"private-{index}", index != 2)

        summary = store.summary()

        self.assertEqual(summary["finished"], 3)
        self.assertEqual(summary["kinds"]["chat"]["elapsed_ms"], {
            "p50": 200.0, "p95": 400.0, "max": 400.0,
        })
        self.assertEqual(summary["kinds"]["chat"]["budget_met_rate"], 0.6667)
        self.assertEqual(summary["kinds"]["chat"]["backends"], {"local-qwen": 3})
        self.assertNotIn("private", str(summary))


if __name__ == "__main__": unittest.main()
