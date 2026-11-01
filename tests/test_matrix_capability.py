import unittest
from unittest.mock import patch

from capability.matrix import impl
from runtime.errors import SchemaValidationError
from runtime.schema import validate


class MatrixCapabilityTests(unittest.TestCase):
    @patch("capability.matrix.impl._call", return_value={"lit_pixels": 1})
    def test_icon_produces_exact_bounded_frame(self, call):
        result = impl.icon({"name": "check", "brightness": 5})
        frame = call.call_args.args[0]["pixels"]
        self.assertEqual(len(frame), 104)
        self.assertEqual(set(frame), {0, 5})
        self.assertEqual(result["icon"], "check")

    @patch("capability.matrix.impl._call", return_value={"lit_pixels": 0})
    def test_clear_sends_black_frame(self, call):
        self.assertEqual(impl.clear(), {"cleared": True})
        self.assertEqual(call.call_args.args[0]["pixels"], [0] * 104)

    def test_frame_schema_enforces_exact_pixel_count(self):
        schema = {"type": "array", "minItems": 104, "maxItems": 104, "items": {"type": "integer"}}
        validate(schema, [0] * 104)
        with self.assertRaises(SchemaValidationError):
            validate(schema, [0] * 103)
        with self.assertRaises(SchemaValidationError):
            validate(schema, [0] * 105)

    @patch("capability.matrix.impl.time.sleep")
    @patch("capability.matrix.impl.time.monotonic", side_effect=[0.0, 0.0, 0.4, 0.5, 0.9, 1.0, 1.0])
    @patch("capability.matrix.impl.clear")
    @patch("capability.matrix.impl.icon")
    def test_blink_is_bounded_and_selects_final_state(self, icon, clear, _clock, _sleep):
        result = impl.blink({
            "name": "heart", "brightness": 5, "duration_seconds": 1,
            "interval_ms": 500, "end_state": "on",
        })
        self.assertEqual(result, {"blinked": True, "icon": "heart", "cycles": 1, "end_state": "on"})
        self.assertEqual(icon.call_count, 2)
        clear.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
