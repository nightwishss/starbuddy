#!/usr/bin/env python3

import json
import unittest

from gateway.llm.adapter import (
    completion_to_sse,
    normalize_tool_arguments,
    prepare_upstream,
    promote_strict_content_tool_calls,
    reject_unpromoted_tool_output,
)


class LLMAdapterTests(unittest.TestCase):
    def test_strips_tools_only_from_continuation(self):
        tool = {"function": {"name": "runtime_call_capability"}}
        initial = {"messages": [{"role": "user", "content": "status"}], "tools": [tool], "stream": True}
        self.assertNotIn("tools", prepare_upstream(initial))
        continuation = {
            "messages": [{"role": "user", "content": "status"}, {"role": "tool", "content": "{}"}],
            "tools": [tool],
            "tool_choice": "auto",
            "stream": True,
        }
        prepared = prepare_upstream(continuation)
        self.assertNotIn("tools", prepared)
        self.assertNotIn("tool_choice", prepared)
        self.assertFalse(prepared["stream"])
        self.assertEqual(prepared["max_completion_tokens"], 128)
        self.assertEqual(prepared["messages"][-1]["role"], "user")
        self.assertIn("工具调用返回", prepared["messages"][-1]["content"])

    def test_caps_hermes_completion_budget(self):
        prepared = prepare_upstream({"messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 65536})
        self.assertEqual(prepared["max_completion_tokens"], 512)

    def test_compact_prompt_names_the_only_allowed_capability(self):
        tool = {"function": {"name": "runtime_call_capability"}}
        prepared = prepare_upstream({"messages": [{"role": "user", "content": "status"}], "tools": [tool]})
        self.assertIn('"capability":"system.status"', prepared["messages"][0]["content"])

    def test_repairs_only_strict_empty_argument_schema(self):
        request = {
            "tools": [{"type": "function", "function": {"name": "status", "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}]
        }
        response = {"choices": [{"message": {"tool_calls": [{"function": {"name": "status", "arguments": '{"type":"object"}'}}]}}]}
        normalized = normalize_tool_arguments(response, request)
        self.assertEqual(normalized["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"], "{}")

    def test_does_not_repair_nonempty_schema(self):
        request = {
            "tools": [{"type": "function", "function": {"name": "call", "parameters": {"type": "object", "properties": {"name": {"type": "string"}}}}}]
        }
        response = {"choices": [{"message": {"tool_calls": [{"function": {"name": "call", "arguments": "{}"}}]}}]}
        self.assertEqual(normalize_tool_arguments(response, request)["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"], "{}")

    def test_sse_contains_tool_call_and_done(self):
        response = {"id": "chat-1", "created": 1, "model": "qwen", "choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "status", "arguments": "{}"}}]}}]}
        body = completion_to_sse(response).decode()
        self.assertIn('"tool_calls"', body)
        self.assertTrue(body.endswith("data: [DONE]\n\n"))
        first = json.loads(body.split("\n\n", 1)[0].removeprefix("data: "))
        self.assertEqual(first["choices"][0]["delta"]["tool_calls"][0]["index"], 0)

    def test_promotes_exact_allowlisted_content_envelope(self):
        request = {"tools": [{"function": {"name": "runtime_call_capability"}}]}
        response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"tool_calls": [{"type": "function", "function": {"name": "runtime_call_capability", "arguments": {"capability": "system.status", "arguments": {}}}}]})}}]}
        promoted = promote_strict_content_tool_calls(response, request)
        choice = promoted["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "runtime_call_capability")

    def test_rejects_unlisted_content_tool(self):
        request = {"tools": [{"function": {"name": "safe"}}]}
        response = {"choices": [{"finish_reason": "stop", "message": {"content": '{"tool_calls":[{"function":{"name":"unsafe","arguments":{}}}]}'}}]}
        promoted = promote_strict_content_tool_calls(response, request)
        self.assertIsNone(promoted["choices"][0]["message"].get("tool_calls"))

    def test_rejects_non_allowlisted_runtime_capability(self):
        wire_name = "mcp_vq_runtime_runtime_call_capability"
        request = {"tools": [{"function": {"name": wire_name}}]}
        response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"tool_calls": [{"function": {"name": wire_name, "arguments": {"capability": "file.write", "arguments": {}}}}]})}}]}
        promoted = promote_strict_content_tool_calls(response, request)
        self.assertIsNone(promoted["choices"][0]["message"].get("tool_calls"))

    def test_malformed_tool_text_cannot_be_returned_as_answer(self):
        request = {"tools": [{"function": {"name": "runtime_call_capability"}}]}
        response = {"choices": [{"message": {"content": '{"tool_calls":[{"function":'}}]}
        with self.assertRaises(ValueError):
            reject_unpromoted_tool_output(response, request)

    def test_repairs_only_known_extra_envelope_suffix(self):
        wire_name = "mcp_vq_runtime_runtime_call_capability"
        request = {"tools": [{"function": {"name": wire_name}}]}
        valid = '{"tool_calls":[{"function":{"name":"' + wire_name + '","arguments":{"capability":"system.status","arguments":{}}}}]}'
        response = {"choices": [{"finish_reason": "stop", "message": {"content": valid + "]}"}}]}
        promoted = promote_strict_content_tool_calls(response, request)
        self.assertEqual(promoted["choices"][0]["finish_reason"], "tool_calls")

    def test_fallback_allowlist_promotes_vq_runtime_tool(self):
        response = {"choices": [{"finish_reason": "stop", "message": {"content": '{"tool_calls":[{"function":{"name":"runtime_call_capability","arguments":{"capability":"system.status","arguments":{}}}}]}'}}]}
        promoted = promote_strict_content_tool_calls(response, {})
        self.assertEqual(promoted["choices"][0]["finish_reason"], "tool_calls")

    def test_maps_short_mcp_name_to_hermes_wire_name(self):
        request = {"tools": [{"function": {"name": "mcp_vq_runtime_runtime_call_capability"}}]}
        response = {"choices": [{"finish_reason": "stop", "message": {"content": '{"tool_calls":[{"function":{"name":"runtime_call_capability","arguments":{"capability":"system.status","arguments":{}}}}]}'}}]}
        promoted = promote_strict_content_tool_calls(response, request)
        name = promoted["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        self.assertEqual(name, "mcp_vq_runtime_runtime_call_capability")

    def test_repairs_known_parameters_field_confusion(self):
        wire_name = "mcp_vq_runtime_runtime_call_capability"
        request = {"tools": [{"function": {"name": wire_name}}]}
        response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"tool_calls": [{"function": {"name": wire_name, "parameters": {"capability": "system.status", "arguments": {}}}}]})}}]}
        promoted = promote_strict_content_tool_calls(response, request)
        function = promoted["choices"][0]["message"]["tool_calls"][0]["function"]
        self.assertEqual(json.loads(function["arguments"]), {"capability": "system.status", "arguments": {}})

    def test_repairs_one_missing_outer_call_brace_then_revalidates(self):
        wire_name = "mcp_vq_runtime_runtime_call_capability"
        request = {"tools": [{"function": {"name": wire_name}}]}
        malformed = '{"tool_calls":[{"type":"function","function":{"name":"' + wire_name + '","arguments":{"capability":"system.status","arguments":{}}}]}'
        response = {"choices": [{"finish_reason": "stop", "message": {"content": malformed}}]}
        promoted = promote_strict_content_tool_calls(response, request)
        self.assertEqual(promoted["choices"][0]["finish_reason"], "tool_calls")

    def test_missing_brace_repair_still_rejects_denied_capability(self):
        wire_name = "mcp_vq_runtime_runtime_call_capability"
        request = {"tools": [{"function": {"name": wire_name}}]}
        malformed = '{"tool_calls":[{"function":{"name":"' + wire_name + '","arguments":{"capability":"file.write","arguments":{}}}]}'
        response = {"choices": [{"finish_reason": "stop", "message": {"content": malformed}}]}
        promoted = promote_strict_content_tool_calls(response, request)
        self.assertIsNone(promoted["choices"][0]["message"].get("tool_calls"))


if __name__ == "__main__":
    unittest.main()
