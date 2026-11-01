"""Mock-only public-situation MCP hook tests; no HTTP, models or devices."""

import asyncio
from copy import deepcopy
import json
from urllib.error import HTTPError
import unittest
from unittest.mock import patch

from integrations.hermes import server as adapter
from interaction_context.situation import TASKS


class SituatedMcpTests(unittest.TestCase):
    def payload(self):
        return {"ok": True, "state": {"revision": 4, "state": "active", "phase": "idle",
                "reason": "decision_recorded", "task": {"id": TASKS[0]["id"]},
                "turn_pending": False, "expires_in_s": 899.5, "offers_enabled": False,
                "attempts": {"correct": 0, "incorrect": 2, "recent": [False, False]},
                "offers": {"enabled": False, "declined": False, "offered": False,
                           "eligible": False, "reason": "disabled"},
                "decision": {"action": "respond", "intent": "help_task", "reason": "model_decision",
                             "target": "苹果加法"},
                "privacy": {"persistent": False, "profile_read": False,
                            "automatic_capture": False, "speaker_binding": "unconfirmed"},
                "timing_ms": {"reasoning": 1200.25, "context": 30.0,
                              "total_to_output_queue": 1250.0, "model_calls": 1}},
                "backend": {"available": True, "backend": "local-bounded", "busy": False,
                    "framework": "hermes", "hermes_available": False, "framework_enabled": False,
                    "framework_blocker": "context_window_unverified_and_private_bootstrap",
                    "automatic_tools": False, "memory_access": False, "cloud_fallback": False,
                    "max_model_calls": 1, "deadline_enforcement": "discard_late_results"}}

    def test_live_catalog_contains_honest_firstclass_readonly_hook(self):
        by_name = {tool.name: tool for tool in adapter.TOOLS}
        tool = by_name[adapter.SITUATION_READ_TOOL]
        self.assertEqual({"type": "object", "properties": {}, "additionalProperties": False}, tool.inputSchema)
        self.assertTrue(tool.annotations.readOnlyHint)
        self.assertFalse(tool.annotations.destructiveHint)
        self.assertTrue(tool.annotations.idempotentHint)
        self.assertFalse(tool.annotations.openWorldHint)
        self.assertIn("不表示Hermes推理循环已启用", tool.description)
        self.assertNotIn(adapter.SITUATION_READ_TOOL, adapter.TOOL_ROUTES)

    def test_runtime_only_catalog_contract_is_preserved(self):
        tools, routes = adapter.build_tool_catalog(frozenset({"system.status"}))
        self.assertEqual({adapter.LEGACY_LIST_TOOL, adapter.LEGACY_CALL_TOOL, "system_status"},
                         {tool.name for tool in tools})
        self.assertEqual({"system_status": "system.status"}, routes)
        tools, _ = adapter.build_tool_catalog(frozenset({"system.status"}), include_situation=True)
        self.assertIn(adapter.SITUATION_READ_TOOL, {tool.name for tool in tools})

    def test_projection_returns_only_trusted_task_and_metadata(self):
        result = adapter._public_situation(self.payload())
        self.assertTrue(result["ok"])
        self.assertFalse(result["raw_inputs_included"])
        self.assertEqual("public_situation_metadata_only", result["scope"])
        self.assertEqual({key: TASKS[0][key] for key in ("id", "title", "prompt")}, result["state"]["task"])
        self.assertEqual(TASKS[0]["id"], result["state"]["decision"]["target"])
        self.assertFalse(result["backend"]["framework_enabled"])
        self.assertEqual(1, result["backend"]["max_model_calls"])

    def test_arbitrary_text_identity_tokens_and_private_fields_are_dropped_recursively(self):
        source = self.payload()
        sentinel = "SYNTHETIC_PRIVATE_SENTINEL"
        source.update(user_input=sentinel, transcript=sentinel, profile={"name": sentinel})
        source["state"].update(raw_input=sentinel, response=sentinel, transcript=sentinel,
                               identity={"label": sentinel}, actors={"speaker": sentinel},
                               token=sentinel, session_id=sentinel, messages=[sentinel],
                               source_states={sentinel: sentinel})
        source["state"]["task"].update(title=sentinel, prompt=sentinel, answer=sentinel, hint=sentinel)
        source["state"]["decision"].update(response=sentinel, target=sentinel, identity=sentinel)
        source["state"]["attempts"].update(raw_answer=sentinel)
        source["backend"].update(error=sentinel, memory={"name": sentinel}, model_path=sentinel)
        dumped = json.dumps(adapter._public_situation(source), ensure_ascii=False)
        self.assertNotIn(sentinel, dumped)
        self.assertNotIn('"answer"', dumped)

    def test_unknown_task_never_forwards_untrusted_task_fields(self):
        source = self.payload()
        source["state"]["task"] = {"id": "unknown", "title": "private", "prompt": "private"}
        result = adapter._public_situation(source)
        self.assertIsNone(result["state"]["task"])
        self.assertNotIn("target", result["state"]["decision"])

    def test_nonfinite_huge_wrong_typed_and_free_text_values_are_dropped(self):
        source = self.payload()
        source["state"].update(revision=True, phase="private", reason="private", expires_in_s=float("nan"))
        source["state"]["task"] = {"id": []}
        source["state"]["attempts"] = {"correct": 10 ** 10000, "incorrect": False,
                                       "recent": ["private"] * 100 + [False, True]}
        source["state"]["offers"] = {"enabled": "true", "reason": "private"}
        source["state"]["decision"] = {"action": "private", "intent": "private", "reason": "private"}
        source["state"]["timing_ms"] = {"context": float("inf"), "reasoning": -1,
                                             "total_to_output_queue": True, "model_calls": 10000}
        source["backend"].update(backend="private", framework="private", framework_blocker="private")
        result = adapter._public_situation(source)
        for name in ("revision", "phase", "reason", "expires_in_s"):
            self.assertNotIn(name, result["state"])
        self.assertEqual({"recent": [False, True]}, result["state"]["attempts"])
        for name in ("offers", "decision", "timing_ms"):
            self.assertEqual({}, result["state"][name])
        self.assertNotIn("private", json.dumps(result))

    def test_projection_size_bounded_even_for_huge_allowed_input_containers(self):
        source = self.payload()
        source["state"]["attempts"]["recent"] = [True] * 100000
        source["state"]["task"]["prompt"] = "x" * 100000
        source["backend"]["error"] = "x" * 100000
        result = adapter._public_situation(source)
        self.assertEqual(8, len(result["state"]["attempts"]["recent"]))
        self.assertLess(len(json.dumps(result, ensure_ascii=False)), 4000)

    def test_projection_does_not_mutate_source(self):
        source = self.payload()
        before = deepcopy(source)
        adapter._public_situation(source)
        self.assertEqual(before, source)

    def test_failed_or_missing_state_fails_closed(self):
        for source in (None, [], {}, {"ok": False, "state": {}}, {"ok": 1, "state": {}},
                       {"ok": True, "state": []}):
            with self.assertRaisesRegex(ValueError, "situation_response_invalid"):
                adapter._public_situation(source)

    @patch("integrations.hermes.server.build_opener")
    def test_read_uses_exact_fixed_get_timeout_and_bounded_read(self, build):
        response = build.return_value.open.return_value.__enter__.return_value
        response.status = 200
        response.read.return_value = json.dumps(self.payload()).encode()
        result = adapter.read_situation()
        build.return_value.open.assert_called_once_with("http://127.0.0.1:8765/api/agent/state", timeout=2.0)
        response.read.assert_called_once_with(65537)
        handlers = build.call_args.args
        self.assertEqual({}, handlers[0].proxies)
        self.assertIsInstance(handlers[1], adapter._NoSituationRedirect)
        self.assertTrue(result["ok"])

    def test_redirect_handler_rejects_every_destination(self):
        handler = adapter._NoSituationRedirect()
        for target in ("http://example.invalid/private", "http://127.0.0.1:8765/api/chat",
                       "file:///private/synthetic", "https://example.invalid"):
            self.assertIsNone(handler.redirect_request(None, None, 302, "Found", {}, target))

    @patch("integrations.hermes.server.build_opener")
    def test_http_network_oversize_or_json_error_has_generic_public_error(self, build):
        response = build.return_value.open.return_value.__enter__.return_value
        for status, body in ((200, b"x" * 65537), (200, b"private invalid json"),
                             (200, b'{"ok":false,"error":"private"}'), (500, b"private")):
            response.status = status
            response.read.return_value = body
            with self.assertRaisesRegex(RuntimeError, "^situation_read_unavailable$"):
                adapter.read_situation()
        build.return_value.open.side_effect = HTTPError("http://private", 302, "private", {}, None)
        with self.assertRaisesRegex(RuntimeError, "^situation_read_unavailable$"):
            adapter.read_situation()

    @patch("integrations.hermes.server.RuntimeClient")
    @patch("integrations.hermes.server.read_situation")
    def test_dispatch_only_reads_public_hook_not_runtime(self, read, runtime):
        read.return_value = {"ok": True, "state": {}}
        result = asyncio.run(adapter.dispatch_tool(adapter.SITUATION_READ_TOOL, {}))
        self.assertEqual(read.return_value, result)
        read.assert_called_once_with()
        runtime.assert_not_called()

    @patch("integrations.hermes.server.read_situation")
    def test_dispatch_rejects_arguments_before_any_io(self, read):
        for arguments in ({"url": "http://example.invalid"}, {"task_id": TASKS[0]["id"]}, None, []):
            with self.assertRaisesRegex(ValueError, "situation_read_accepts_no_arguments"):
                asyncio.run(adapter.dispatch_tool(adapter.SITUATION_READ_TOOL, arguments))
        read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
