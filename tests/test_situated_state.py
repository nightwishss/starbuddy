"""Synthetic, pure state tests: no device, model, media or personal data."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import unittest

from interaction_context.situation import SituationState, TASKS


class SituationStateTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        self.state = SituationState(clock=lambda: self.now)
        self.task_id = TASKS[0]["id"]

    def select(self):
        return self.state.select(self.task_id)

    def eligible(self):
        self.select()
        self.state.set_offers(True)
        self.state.feedback(False)
        return self.state.feedback(False)

    def test_catalogue_has_three_small_stable_json_tasks(self):
        self.assertEqual(3, len(TASKS))
        self.assertEqual(3, len({task["id"] for task in TASKS}))
        self.assertLess(len(json.dumps(TASKS, ensure_ascii=False)), 4000)
        for task in TASKS:
            self.assertEqual({"id", "title", "prompt", "answer", "hint", "steps"}, set(task))
            self.assertTrue(all(isinstance(step, str) for step in task["steps"]))

    def test_initial_state_is_idle_and_proactive_help_is_opt_in(self):
        snapshot = self.state.snapshot()
        self.assertIsNone(snapshot["task"])
        self.assertEqual("idle", snapshot["state"])
        self.assertFalse(snapshot["offers"]["enabled"])
        self.assertFalse(snapshot["turn_pending"])

    def test_select_exposes_prompt_not_answer_and_increments_revision(self):
        before = self.state.snapshot()["revision"]
        selected = self.select()
        self.assertGreater(selected["revision"], before)
        self.assertEqual(self.task_id, selected["task"]["id"])
        self.assertEqual({"id", "title", "prompt"}, set(selected["task"]))
        self.assertEqual("active", selected["state"])

    def test_unknown_or_non_string_task_ids_do_not_change_state(self):
        self.select()
        before = self.state.snapshot()
        for value in ("unknown", "", "x" * 100000, None, True, [], {}):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(ValueError):
                    self.state.select(value)
        self.assertEqual(before, self.state.snapshot())

    def test_snapshot_and_packet_are_independent_copies(self):
        self.select()
        snapshot = self.state.snapshot()
        snapshot["task"]["prompt"] = "rewritten"
        snapshot["attempts"]["recent"].append(True)
        snapshot["offers"]["enabled"] = True
        ticket = self.state.begin("text")
        packet = self.state.packet(ticket)
        packet["task"]["steps"].append("rewritten")
        packet["actors"]["microphone_speaker"]["profile_authorized"] = True
        self.assertEqual(TASKS[0]["prompt"], self.state.snapshot()["task"]["prompt"])
        self.assertEqual([], self.state.snapshot()["attempts"]["recent"])
        self.assertEqual(3, len(self.state.packet(ticket)["task"]["steps"]))
        self.assertFalse(self.state.packet(ticket)["actors"]["microphone_speaker"]["profile_authorized"])

    def test_public_catalogue_mutation_does_not_replace_authoritative_task(self):
        original = deepcopy(TASKS[0])
        try:
            TASKS[0]["prompt"] = "untrusted replacement"
            TASKS[0]["answer"] = "wrong"
            TASKS[0]["steps"].append("untrusted")
            self.select()
            packet = self.state.packet(self.state.begin("text"))
            self.assertEqual(original, packet["task"])
        finally:
            TASKS[0].clear()
            TASKS[0].update(original)

    def test_packet_distinguishes_actors_without_profile_authority(self):
        for input_type in ("voice", "touch", "text"):
            self.select()
            packet = self.state.packet(self.state.begin(input_type))
            actors = packet["actors"]
            self.assertEqual("unproven", actors["cross_modal_identity_link"])
            self.assertEqual("unknown", actors["microphone_speaker"]["identity"])
            self.assertEqual("unknown", actors["touch_operator"]["identity"])
            self.assertFalse(actors["microphone_speaker"]["profile_authorized"])
            self.assertFalse(actors["touch_operator"]["profile_authorized"])
            self.assertFalse(actors["visible_person"]["profile_authorized"])

    def test_begin_replaces_earlier_turn_even_for_same_task(self):
        self.select()
        first = self.state.begin("voice")
        second = self.state.begin("text")
        self.assertFalse(self.state.current(first))
        self.assertIsNone(self.state.packet(first))
        self.assertTrue(self.state.current(second))
        self.assertGreater(second["revision"], first["revision"])

    def test_begin_without_task_does_not_invent_task(self):
        ticket = self.state.begin("voice")
        self.assertTrue(self.state.current(ticket))
        self.assertIsNone(self.state.packet(ticket)["task"])
        self.assertIsNone(self.state.packet(ticket)["task_source"])

    def test_invalid_input_modes_are_rejected_without_turn_replacement(self):
        ticket = self.state.begin("text")
        for value in ("camera", "", "VOICE", None, [], True):
            with self.assertRaises(ValueError):
                self.state.begin(value)
        self.assertTrue(self.state.current(ticket))

    def test_mutated_malformed_or_cross_instance_tickets_are_not_current(self):
        self.select()
        ticket = self.state.begin("text")
        for field, value in (("revision", -1), ("task_id", TASKS[1]["id"]),
                             ("token", "fake"), ("input_type", "voice")):
            tampered = {**ticket, field: value}
            self.assertFalse(self.state.current(tampered))
            self.assertFalse(self.state.finish(tampered, {"action": "answer"}))
        for invalid in (None, [], {}, True, "ticket"):
            self.assertFalse(self.state.current(invalid))
        other = SituationState(clock=lambda: self.now)
        other.select(self.task_id)
        self.assertFalse(self.state.current(other.begin("text")))
        self.assertTrue(self.state.current(ticket))

    def test_select_clear_cancel_feedback_and_decline_revoke_old_turn(self):
        for operation in (lambda: self.state.select(TASKS[1]["id"]), lambda: self.state.clear(),
                          lambda: self.state.invalidate(), lambda: self.state.feedback(False),
                          lambda: self.state.decline(), lambda: self.state.set_offers(True)):
            with self.subTest(operation=operation):
                self.state = SituationState(clock=lambda: self.now)
                self.select()
                old = self.state.begin("voice")
                operation()
                after = self.state.snapshot()
                self.assertFalse(self.state.current(old))
                self.assertFalse(self.state.finish(old, {"action": "answer"}))
                self.assertEqual(after, self.state.snapshot())

    def test_cancel_keeps_task_progress_and_preferences(self):
        self.eligible()
        self.state.decline()
        old = self.state.begin("text")
        cancelled = self.state.invalidate()
        self.assertEqual(self.task_id, cancelled["task"]["id"])
        self.assertEqual(2, cancelled["attempts"]["incorrect"])
        self.assertTrue(cancelled["offers"]["enabled"])
        self.assertTrue(cancelled["offers"]["declined"])
        self.assertFalse(self.state.current(old))

    def test_clear_discards_episode_but_retains_explicit_offer_preference(self):
        self.eligible()
        self.state.decline()
        snapshot = self.state.clear()
        self.assertIsNone(snapshot["task"])
        self.assertEqual([], snapshot["attempts"]["recent"])
        self.assertEqual(0, snapshot["attempts"]["incorrect"])
        self.assertFalse(snapshot["offers"]["declined"])
        self.assertTrue(snapshot["offers"]["enabled"])

    def test_offer_requires_enabled_two_incorrect_and_only_once_per_task(self):
        self.select()
        self.state.feedback(False)
        self.state.feedback(False)
        ticket = self.state.begin("text")
        self.assertFalse(self.state.finish(ticket, {"action": "offer_help"}))
        self.state.set_offers(True)
        ticket = self.state.begin("text")
        self.assertTrue(self.state.snapshot()["offers"]["eligible"])
        self.assertTrue(self.state.finish(ticket, {"action": "offer_help", "reason": "offer_eligible"}))
        self.assertTrue(self.state.snapshot()["offers"]["offered"])
        next_ticket = self.state.begin("text")
        self.assertFalse(self.state.finish(next_ticket, {"action": "offer_help"}))

    def test_one_incorrect_never_makes_offer_eligible(self):
        self.select()
        self.state.set_offers(True)
        self.state.feedback(False)
        self.assertFalse(self.state.snapshot()["offers"]["eligible"])

    def test_decline_latches_across_errors_turns_and_preference_toggles(self):
        self.eligible()
        self.state.decline()
        self.state.feedback(False)
        self.state.set_offers(False)
        self.state.set_offers(True)
        ticket = self.state.begin("text")
        self.assertFalse(self.state.snapshot()["offers"]["eligible"])
        self.assertFalse(self.state.finish(ticket, {"action": "offer_help"}))
        # Explicitly asking for help is still permitted; decline is not a ban on answering.
        self.assertTrue(self.state.finish(ticket, {"action": "answer", "intent": "task_help"}))
        self.assertTrue(self.state.snapshot()["offers"]["declined"])

    def test_new_explicit_task_episode_resets_decline_and_attempts(self):
        self.eligible()
        self.state.decline()
        snapshot = self.select()
        self.assertFalse(snapshot["offers"]["declined"])
        self.assertFalse(snapshot["offers"]["offered"])
        self.assertEqual(0, snapshot["attempts"]["incorrect"])

    def test_correct_feedback_completes_and_suppresses_further_offers(self):
        self.eligible()
        completed = self.state.feedback(True)
        self.assertEqual("completed", completed["state"])
        self.assertEqual("task_complete", completed["offers"]["reason"])
        self.assertEqual(completed, self.state.feedback(False))
        self.assertEqual(completed, self.state.feedback(True))

    def test_feedback_without_task_does_not_invent_progress(self):
        before = self.state.snapshot()
        self.assertEqual(before, self.state.feedback(False))

    def test_booleans_are_strict(self):
        for value in (0, 1, "true", "false", None, [], {}):
            with self.assertRaises(TypeError):
                self.state.feedback(value)
            with self.assertRaises(TypeError):
                self.state.set_offers(value)

    def test_finish_is_single_use_and_never_stores_text_or_identity(self):
        self.select()
        ticket = self.state.begin("voice")
        decision = {"action": "answer", "intent": "task_help", "reason": "explicit_request",
                    "needs_clarification": False, "text": "synthetic secret phrase",
                    "identity": "synthetic identity", "profile_authorized": True,
                    "long": "x" * 100000}
        self.assertTrue(self.state.finish(ticket, decision))
        self.assertFalse(self.state.finish(ticket, decision))
        self.assertFalse(self.state.current(ticket))
        record = self.state.snapshot()["last_decision"]
        self.assertEqual({"action": "answer", "intent": "task_help", "reason": "explicit_request",
                          "needs_clarification": False}, record)
        dumped = json.dumps(self.state.snapshot())
        self.assertNotIn("synthetic secret", dumped)
        self.assertNotIn("synthetic identity", dumped)

    def test_bad_decisions_cannot_complete_task_or_consume_ticket(self):
        self.select()
        ticket = self.state.begin("text")
        for decision in (None, [], {}, {"action": "x" * 100000}, {"action": []}):
            self.assertFalse(self.state.finish(ticket, decision))
            self.assertTrue(self.state.current(ticket))
        self.assertTrue(self.state.finish(ticket, {"action": "complete", "correct": True,
                                                  "intent": "private-identity", "reason": "x" * 1000}))
        self.assertEqual("active", self.state.snapshot()["state"])
        self.assertEqual({"action": "complete"}, self.state.snapshot()["last_decision"])

    def test_recent_outcomes_and_counter_memory_are_bounded(self):
        self.select()
        for _ in range(1200):
            self.state.feedback(False)
        snapshot = self.state.snapshot()
        packet = self.state.packet(self.state.begin("text"))
        self.assertEqual(8, len(snapshot["attempts"]["recent"]))
        self.assertEqual(999, snapshot["attempts"]["incorrect"])
        self.assertLess(len(json.dumps(snapshot, ensure_ascii=False)), 3000)
        self.assertLess(len(json.dumps(packet, ensure_ascii=False)), 4000)

    def test_ticket_expires_at_boundary_and_never_revives(self):
        self.select()
        ticket = self.state.begin("text")
        self.now += 59.999
        self.assertTrue(self.state.current(ticket))
        self.now += 0.001
        self.assertFalse(self.state.current(ticket))
        self.assertIsNone(self.state.packet(ticket))
        self.assertEqual(self.task_id, self.state.snapshot()["task"]["id"])
        self.assertFalse(self.state.finish(ticket, {"action": "answer"}))

    def test_task_expires_without_polling_extending_session(self):
        self.select()
        for _ in range(8):
            self.now += 100
            self.assertIsNotNone(self.state.snapshot()["task"])
        self.now += 100
        expired = self.state.snapshot()
        self.assertIsNone(expired["task"])
        self.assertEqual("task_expired", expired["reason"])
        self.assertIsNone(self.state.packet(self.state.begin("text"))["task"])

    def test_real_user_activity_extends_task_but_not_old_ticket(self):
        self.select()
        old = self.state.begin("text")
        self.now += 850
        self.state.feedback(False)
        self.now += 850
        self.assertIsNotNone(self.state.snapshot()["task"])
        self.assertFalse(self.state.current(old))
        self.now += 50
        self.assertIsNone(self.state.snapshot()["task"])

    def test_task_expiration_revokes_inflight_turn(self):
        short = SituationState(clock=lambda: self.now, inactivity_ttl=10, ticket_ttl=60)
        short.select(self.task_id)
        ticket = short.begin("text")
        self.now += 10
        self.assertFalse(short.current(ticket))
        self.assertFalse(short.finish(ticket, {"action": "answer"}))
        self.assertIsNone(short.snapshot()["task"])

    def test_clock_rollback_cannot_revive_task_or_old_ticket(self):
        self.select()
        ticket = self.state.begin("voice")
        self.now -= 1
        self.assertFalse(self.state.current(ticket))
        self.assertIsNone(self.state.snapshot()["task"])
        self.assertEqual("clock_reset", self.state.snapshot()["reason"])

    def test_bad_clock_fails_closed(self):
        self.select()
        ticket = self.state.begin("voice")
        self.now = float("nan")
        with self.assertRaises(ValueError):
            self.state.current(ticket)
        self.now = 1001.0
        self.assertIsNone(self.state.snapshot()["task"])
        self.assertFalse(self.state.current(ticket))

    def test_ttl_configuration_cannot_disable_bounded_lifetimes(self):
        for value in (0, -1, float("inf"), float("nan"), "900", True, 901):
            with self.assertRaises(ValueError):
                SituationState(inactivity_ttl=value)
        with self.assertRaises(ValueError):
            SituationState(ticket_ttl=61)

    def test_concurrent_completion_can_commit_only_once(self):
        self.select()
        ticket = self.state.begin("text")
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.state.finish(ticket, {"action": "answer"}), range(32)))
        self.assertEqual(1, sum(results))

    def test_concurrent_new_turns_leave_exactly_one_current_ticket(self):
        self.select()
        with ThreadPoolExecutor(max_workers=8) as pool:
            tickets = list(pool.map(lambda _: self.state.begin("text"), range(32)))
        self.assertEqual(1, sum(self.state.current(ticket) for ticket in tickets))
        self.assertEqual(32, len({ticket["token"] for ticket in tickets}))


if __name__ == "__main__":
    unittest.main()
