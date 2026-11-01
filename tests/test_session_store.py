#!/usr/bin/env python3

import unittest

from orchestration import Affect, InteractionEvent, SessionStore
from governance import DataLabel, PrivacyContext


class SessionStoreTests(unittest.TestCase):
    def test_context_is_bounded_and_copied(self):
        store = SessionStore(max_turns=2)
        for index in range(3):
            store.append_turn("phone", f"u{index}", f"b{index}")
        context = store.context("phone")
        self.assertEqual([item["text"] for item in context], ["u1", "b1", "u2", "b2"])
        context[0]["text"] = "changed"
        self.assertEqual(store.context("phone")[0]["text"], "u1")

    def test_interaction_trace_spans_one_turn(self):
        store = SessionStore()
        first = store.transition("phone", InteractionEvent.LISTEN_STARTED)
        second = store.transition("phone", InteractionEvent.INPUT_READY)
        third = store.transition("phone", InteractionEvent.RESPONSE_READY)
        final = store.transition("phone", InteractionEvent.SPEECH_FINISHED)
        next_turn = store.transition("phone", InteractionEvent.INPUT_READY)
        self.assertEqual(len({first[0], second[0], third[0], final[0]}), 1)
        self.assertNotEqual(first[0], next_turn[0])

    def test_expiry_lru_and_reset_remove_data(self):
        now = [0.0]
        store = SessionStore(ttl_seconds=5, max_sessions=2, clock=lambda: now[0])
        store.append_turn("a", "u", "b")
        store.context("b")
        store.context("c")
        self.assertEqual(store.stats()["active_sessions"], 2)
        self.assertEqual(store.context("a"), [])
        self.assertTrue(store.reset("a"))
        now[0] = 10
        self.assertEqual(store.stats()["active_sessions"], 0)

    def test_privacy_context_is_session_scoped(self):
        store = SessionStore()
        private = PrivacyContext(frozenset({DataLabel.CHILD_PRIVATE}), local_only=True)
        store.set_privacy("child", private)
        self.assertEqual(store.privacy("child"), private)
        self.assertNotEqual(store.privacy("other"), private)

    def test_affect_updates_expression_without_storing_evidence_text(self):
        store = SessionStore()
        store.update_affect("child", Affect(-0.6, 0.1, 0.8))
        store.transition("child", InteractionEvent.INPUT_READY)
        _, intent, _ = store.transition("child", InteractionEvent.RESPONSE_READY)
        self.assertEqual(intent.visual_style, "gentle")

    def test_direct_idle_pipeline_can_renew_trace(self):
        store = SessionStore()
        first = store.ensure_trace("phone", renew_if_idle=True)
        second = store.ensure_trace("phone", renew_if_idle=True)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
