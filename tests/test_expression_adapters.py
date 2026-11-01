#!/usr/bin/env python3
"""Tests for replaceable expression adapters and failure isolation."""

import unittest
from unittest.mock import Mock

from orchestration import InteractionEvent, InteractionOrchestrator
from orchestration.adapters import CompositeExpressionAdapter, DisplayExpressionAdapter, MatrixExpressionAdapter


class ExpressionAdapterTests(unittest.TestCase):
    def test_matrix_maps_each_phase_through_sdk(self):
        matrix = Mock()
        adapter = MatrixExpressionAdapter(matrix)
        machine = InteractionOrchestrator()

        adapter.apply(machine.transition(InteractionEvent.LISTEN_STARTED), "trace-listen")
        adapter.apply(machine.transition(InteractionEvent.INPUT_READY), "trace-think")
        adapter.apply(machine.transition(InteractionEvent.RESPONSE_READY), "trace-speak")
        adapter.apply(machine.transition(InteractionEvent.SPEECH_FINISHED), "trace-idle")

        matrix.icon.assert_any_call("heart", 4, request_id="trace-listen")
        matrix.draw.assert_called_once_with(adapter._THINKING, request_id="trace-think")
        matrix.icon.assert_any_call("smile", 5, request_id="trace-speak")
        matrix.icon.assert_any_call("smile", 2, request_id="trace-idle")

    def test_composite_isolates_one_failed_device(self):
        good = Mock(name="good")
        good.name = "good"
        good.apply.return_value = "ok"
        bad = Mock(name="bad")
        bad.name = "bad"
        bad.apply.side_effect = RuntimeError("private hardware detail")
        intent = InteractionOrchestrator().intent()

        results = CompositeExpressionAdapter([bad, good]).apply(intent, "trace-1")

        self.assertFalse(results[0].ok)
        self.assertEqual(results[0].detail, "device unavailable")
        self.assertTrue(results[1].ok)
        good.apply.assert_called_once_with(intent, "trace-1")

    def test_display_maps_abstract_phase_without_hardware_detail(self):
        display = Mock()
        adapter = DisplayExpressionAdapter(display)
        intent = InteractionOrchestrator().transition(InteractionEvent.LISTEN_STARTED)

        adapter.apply(intent, "trace-listen")

        display.update.assert_called_once_with(
            phase="listening", visual_style="attentive",
            primary_text="我在听", secondary_text="自然说话就好，说完停一下",
            trace_id="trace-listen",
        )


if __name__ == "__main__":
    unittest.main()
