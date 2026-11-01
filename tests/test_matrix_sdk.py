import unittest
from unittest.mock import Mock

from sdk.matrix import MatrixClient, MatrixError


class MatrixClientTests(unittest.TestCase):
    def test_methods_route_only_through_runtime_contracts(self):
        runtime = Mock()
        runtime.call.side_effect = [
            {"ok": True, "result": {"ready": True, "rows": 8, "columns": 13}},
            {"ok": True, "result": {"drawn": True, "icon": "heart", "lit_pixels": 30}},
            {"ok": True, "result": {"drawn": True, "lit_pixels": 1}},
            {"ok": True, "result": {"blinked": True, "icon": "heart", "cycles": 10, "end_state": "on"}},
            {"ok": True, "result": {"cleared": True}},
        ]
        client = MatrixClient(runtime)

        self.assertTrue(client.status()["ready"])
        self.assertEqual(client.icon("heart", 4)["icon"], "heart")
        self.assertTrue(client.draw([0] * 103 + [7])["drawn"])
        self.assertTrue(client.blink("heart", brightness=4)["blinked"])
        self.assertTrue(client.clear()["cleared"])
        self.assertEqual(runtime.call.call_args_list[0].args, ("matrix.status", {}))
        self.assertEqual(runtime.call.call_args_list[1].args, ("matrix.icon", {"name": "heart", "brightness": 4}))
        self.assertEqual(runtime.call.call_args_list[2].args, ("matrix.draw", {"pixels": [0] * 103 + [7]}))
        self.assertEqual(runtime.call.call_args_list[3].args, ("matrix.blink", {
            "name": "heart", "brightness": 4, "duration_seconds": 10,
            "interval_ms": 500, "end_state": "on",
        }))
        self.assertEqual(runtime.call.call_args_list[4].args, ("matrix.clear", {}))

    def test_expression_trace_is_forwarded_to_runtime(self):
        runtime = Mock()
        runtime.call.return_value = {"ok": True, "result": {"icon": "smile"}}
        MatrixClient(runtime).icon("smile", 2, request_id="interaction-123")
        runtime.call.assert_called_once_with(
            "matrix.icon", {"name": "smile", "brightness": 2}, request_id="interaction-123"
        )

    def test_runtime_error_is_exposed_without_hardware_details(self):
        runtime = Mock()
        runtime.call.return_value = {
            "ok": False,
            "error": {"code": "SCHEMA_VALIDATION_ERROR", "message": "bad brightness", "details": {"path": "brightness"}},
        }
        with self.assertRaises(MatrixError) as caught:
            MatrixClient(runtime).icon("check", 8)
        self.assertEqual(caught.exception.code, "SCHEMA_VALIDATION_ERROR")
        self.assertEqual(caught.exception.details, {"path": "brightness"})


if __name__ == "__main__":
    unittest.main()
