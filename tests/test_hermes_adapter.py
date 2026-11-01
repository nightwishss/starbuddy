"""Unit tests for the Hermes-to-Runtime MCP adapter seam."""

import unittest
from unittest.mock import patch

from integrations.hermes.server import (
    LEGACY_CALL_TOOL,
    LEGACY_LIST_TOOL,
    TOOL_ROUTES,
    build_tool_catalog,
    call_runtime,
    capability_catalog,
    capability_tool_name,
    ensure_capability_allowed,
)


class HermesAdapterTests(unittest.TestCase):
    def test_catalog_has_only_public_metadata(self):
        catalog = capability_catalog()
        self.assertGreater(len(catalog), 0)
        self.assertIn("system.info", {entry["name"] for entry in catalog})
        self.assertNotIn("audio.capture_clip", {entry["name"] for entry in catalog})
        self.assertTrue(all("implementation" not in entry for entry in catalog))

    def test_capability_names_are_direct_tools(self):
        self.assertEqual(capability_tool_name("system.status"), "system_status")
        self.assertEqual(TOOL_ROUTES["system_status"], "system.status")
        self.assertEqual(TOOL_ROUTES["led_set_rgb"], "led.set_rgb")
        self.assertEqual(TOOL_ROUTES["matrix_status"], "matrix.status")
        self.assertEqual(TOOL_ROUTES["matrix_icon"], "matrix.icon")
        self.assertEqual(TOOL_ROUTES["matrix_blink"], "matrix.blink")
        self.assertEqual(TOOL_ROUTES["matrix_draw"], "matrix.draw")
        self.assertEqual(TOOL_ROUTES["matrix_clear"], "matrix.clear")

    def test_tool_catalog_preserves_runtime_input_schema(self):
        tools, routes = build_tool_catalog()
        by_name = {tool.name: tool for tool in tools}
        self.assertEqual(len(by_name), len(routes) + 2)
        self.assertIn(LEGACY_LIST_TOOL, by_name)
        self.assertIn(LEGACY_CALL_TOOL, by_name)
        schema = by_name["led_set_rgb"].inputSchema
        self.assertEqual(schema["required"], ["red", "green", "blue"])
        self.assertEqual(schema["properties"]["red"]["maximum"], 255)
        matrix_schema = by_name["matrix_draw"].inputSchema
        self.assertEqual(matrix_schema["properties"]["pixels"]["minItems"], 104)
        self.assertEqual(matrix_schema["properties"]["pixels"]["maxItems"], 104)

    def test_query_tools_have_read_only_annotations(self):
        tools, _routes = build_tool_catalog()
        by_name = {tool.name: tool for tool in tools}
        self.assertTrue(by_name["system_status"].annotations.readOnlyHint)
        self.assertFalse(by_name["system_status"].annotations.openWorldHint)

    def test_allowlisted_catalog_exposes_only_selected_capability(self):
        tools, routes = build_tool_catalog(frozenset({"system.status"}))
        self.assertEqual(set(routes.values()), {"system.status"})
        self.assertEqual({tool.name for tool in tools}, {LEGACY_LIST_TOOL, LEGACY_CALL_TOOL, "system_status"})

    @patch("integrations.hermes.server.CAPABILITY_ALLOWLIST", frozenset({"system.status"}))
    def test_allowlist_rejects_other_capability_before_runtime(self):
        ensure_capability_allowed("system.status")
        with self.assertRaisesRegex(RuntimeError, "不允许"):
            ensure_capability_allowed("file.write")

    def test_legacy_generic_tool_cannot_bypass_internal_capability_boundary(self):
        with self.assertRaisesRegex(RuntimeError, "内部Runtime"):
            ensure_capability_allowed("audio.capture_clip")
        with self.assertRaisesRegex(RuntimeError, "内部Runtime"):
            ensure_capability_allowed("speech.transcribe")

    @patch("integrations.hermes.server.RuntimeClient")
    def test_success_response_is_unwrapped(self, client_type):
        client_type.return_value.call.return_value = {
            "ok": True, "result": {"hostname": "ventunoq"},
            "meta": {"elapsed_ms": 1.0},
        }
        self.assertEqual(
            call_runtime("system.info"),
            {"result": {"hostname": "ventunoq"}, "meta": {"elapsed_ms": 1.0}},
        )
        client_type.return_value.call.assert_called_once_with("system.info", {})

    @patch("integrations.hermes.server.RuntimeClient")
    def test_runtime_error_becomes_tool_error(self, client_type):
        client_type.return_value.call.return_value = {
            "ok": False,
            "error": {"code": "permission_denied", "message": "not allowed"},
        }
        with self.assertRaisesRegex(RuntimeError, "permission_denied: not allowed"):
            call_runtime("led.off")


if __name__ == "__main__":
    unittest.main()
