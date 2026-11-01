#!/usr/bin/env python3

import unittest
from pathlib import Path
from unittest.mock import MagicMock, call

from sdk.display import DisplayClient, DisplayError


class DisplaySdkTests(unittest.TestCase):
    def test_session_closes_after_consumer_error(self):
        runtime = MagicMock()
        runtime.call.side_effect = [
            {"ok": True, "result": {"display_id": "display_test"}},
            {"ok": True, "result": {"display_id": "display_test", "closed": True}},
        ]
        with self.assertRaisesRegex(RuntimeError, "render failed"):
            with DisplayClient(runtime).session(width=640, height=480) as display:
                self.assertEqual(display["display_id"], "display_test")
                raise RuntimeError("render failed")
        self.assertEqual(runtime.call.call_args_list, [
            call("display.open", {
                "width": 640, "height": 480, "backend": "framebuffer",
            }, timeout_ms=2_000),
            call("display.close", {"display_id": "display_test"}, timeout_ms=2_000),
        ])

    def test_render_methods_keep_trusted_paths_explicit(self):
        runtime = MagicMock()
        runtime.call.return_value = {"ok": True, "result": {"rendered": True}}
        client = DisplayClient(runtime)
        client.show_text("display_test", "你好", (10, 20))
        client.show_image_path("display_test", Path("/tmp/test.png"))
        self.assertEqual(runtime.call.call_args_list[0], call(
            "display.show_text", {"display_id": "display_test", "text": "你好", "position": [10, 20]},
            timeout_ms=2_000,
        ))
        self.assertEqual(runtime.call.call_args_list[1], call(
            "display.show_image", {"display_id": "display_test", "path": "/tmp/test.png"},
            timeout_ms=3_000,
        ))

    def test_protocol_error_is_typed(self):
        runtime = MagicMock()
        runtime.call.return_value = {"ok": True, "result": None}
        with self.assertRaises(DisplayError) as raised:
            DisplayClient(runtime).list()
        self.assertEqual(raised.exception.code, "PROTOCOL_ERROR")


if __name__ == "__main__":
    unittest.main()
