#!/usr/bin/env python3
"""Tests for the device-independent interaction state machine."""

import unittest

from orchestration import (
    Affect,
    InteractionEvent,
    InteractionOrchestrator,
    InteractionPhase,
    InvalidTransition,
)


class InteractionOrchestratorTests(unittest.TestCase):
    def test_complete_voice_turn_has_deterministic_states(self):
        machine = InteractionOrchestrator()
        phases = [
            machine.transition(InteractionEvent.LISTEN_STARTED).phase,
            machine.transition(InteractionEvent.INPUT_READY).phase,
            machine.transition(InteractionEvent.RESPONSE_READY).phase,
            machine.transition(InteractionEvent.SPEECH_FINISHED).phase,
        ]
        self.assertEqual(
            phases,
            [InteractionPhase.LISTENING, InteractionPhase.THINKING,
             InteractionPhase.SPEAKING, InteractionPhase.IDLE],
        )
        self.assertEqual(machine.snapshot()["sequence"], 4)

    def test_speaking_can_be_interrupted_by_new_input(self):
        machine = InteractionOrchestrator()
        for event in (InteractionEvent.LISTEN_STARTED, InteractionEvent.INPUT_READY,
                      InteractionEvent.RESPONSE_READY):
            machine.transition(event)
        intent = machine.transition(InteractionEvent.LISTEN_STARTED)
        self.assertEqual(intent.phase, InteractionPhase.LISTENING)

    def test_negative_confident_affect_selects_gentle_expression(self):
        machine = InteractionOrchestrator()
        machine.update_affect(Affect(valence=-0.7, arousal=0.2, confidence=0.8))
        for event in (InteractionEvent.LISTEN_STARTED, InteractionEvent.INPUT_READY,
                      InteractionEvent.RESPONSE_READY):
            intent = machine.transition(event)
        self.assertEqual(intent.visual_style, "gentle")
        self.assertLess(intent.speech_rate, 1.0)

    def test_invalid_transitions_and_affect_ranges_are_rejected(self):
        machine = InteractionOrchestrator()
        with self.assertRaises(InvalidTransition):
            machine.transition(InteractionEvent.RESPONSE_READY)
        with self.assertRaises(ValueError):
            Affect(valence=2.0)

    def test_error_requires_reset_or_new_listen(self):
        machine = InteractionOrchestrator()
        self.assertEqual(machine.transition(InteractionEvent.FAILED).phase, InteractionPhase.ERROR)
        self.assertEqual(machine.transition(InteractionEvent.RESET).phase, InteractionPhase.IDLE)


if __name__ == "__main__":
    unittest.main()
