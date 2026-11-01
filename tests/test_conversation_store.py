import threading
import time
import unittest

from conversation_service.store import ConversationError, ConversationStore


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class ConversationStoreTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.store = ConversationStore(
            ttl_seconds=60, max_sessions=2, max_turns=2,
            max_context_chars=40, max_memory_items=2,
            clock=self.clock, wall_clock=self.clock,
        )

    def complete(self, session, user, answer):
        active = self.store.start_turn(session, user)
        turn_id = active["active_turn"]["turn_id"]
        return self.store.complete_turn(session, turn_id, answer, backend="test")

    def test_atomic_turn_and_bounded_context(self):
        self.complete("one", "第一轮", "回答一")
        self.complete("one", "第二轮", "回答二")
        result = self.complete("one", "第三轮", "回答三")
        self.assertEqual([item["text"] for item in result["messages"]],
                         ["第二轮", "回答二", "第三轮", "回答三"])
        self.assertEqual(result["last_turn"]["status"], "completed")
        self.assertIsNone(result["active_turn"])

    def test_only_one_active_turn_and_matching_owner(self):
        result = self.store.start_turn("one", "你好", input_type="voice")
        turn_id = result["active_turn"]["turn_id"]
        with self.assertRaisesRegex(ConversationError, "already active"):
            self.store.start_turn("one", "再次说话")
        with self.assertRaisesRegex(ConversationError, "does not own"):
            self.store.complete_turn("one", "wrong", "回答")
        result = self.store.end_turn("one", turn_id, status="cancelled", reason="barge-in")
        self.assertEqual(result["last_turn"]["status"], "cancelled")
        self.assertEqual(result["messages"], [])

    def test_failed_turn_does_not_pollute_history(self):
        result = self.store.start_turn("one", "模型会失败")
        result = self.store.end_turn("one", result["active_turn"]["turn_id"],
                                     status="failed", reason="timeout")
        self.assertEqual(result["messages"], [])
        self.assertEqual(result["last_turn"]["reason"], "timeout")

    def test_abandoned_active_turn_expires_and_session_recovers(self):
        self.store = ConversationStore(
            ttl_seconds=60, turn_timeout_seconds=5,
            clock=self.clock, wall_clock=self.clock,
        )
        abandoned = self.store.start_turn("one", "不会完成")
        self.assertGreater(abandoned["active_turn"]["expires_in_seconds"], 0)
        self.clock.advance(6)
        recovered = self.store.start_turn("one", "新的输入")
        self.assertEqual(recovered["last_turn"]["reason"], "turn_timeout")
        self.assertEqual(recovered["active_turn"]["user_text"], "新的输入")

    def test_working_memory_replace_expire_forget_and_evict(self):
        self.store.remember("one", "scene", {"person": 1}, source="vision", ttl_seconds=10)
        self.clock.advance(1)
        self.store.remember("one", "mood", "calm", source="interaction")
        self.clock.advance(1)
        result = self.store.remember("one", "scene", {"person": 2}, source="vision")
        self.assertEqual(result["memory"]["scene"]["value"]["person"], 2)
        self.clock.advance(1)
        result = self.store.remember("one", "topic", "robot", source="agent")
        self.assertEqual(set(result["memory"]), {"scene", "topic"})
        result = self.store.forget("one", "scene")
        self.assertTrue(result["forgotten"])
        self.store.remember("one", "brief", True, source="test", ttl_seconds=5)
        self.clock.advance(6)
        self.assertNotIn("brief", self.store.context("one")["memory"])

    def test_ttl_and_lru_bound(self):
        self.store.ensure("one")
        self.store.ensure("two")
        self.store.ensure("three")
        self.assertEqual(self.store.stats()["active_sessions"], 2)
        self.clock.advance(61)
        self.assertEqual(self.store.stats()["active_sessions"], 0)
        recreated = self.store.ensure("three", privacy="local-sensitive")
        self.assertEqual(recreated["privacy"], "local-sensitive")

    def test_wait_wakes_on_revision(self):
        revision = self.store.ensure("one")["revision"]
        holder = {}

        def waiter():
            holder["result"] = self.store.wait("one", revision, timeout=1)

        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.02)
        self.store.start_turn("one", "hello")
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertTrue(holder["result"]["changed"])

    def test_reset_and_validation(self):
        self.store.ensure("one")
        self.assertTrue(self.store.reset("one")["reset"])
        with self.assertRaises(ConversationError):
            self.store.ensure("contains spaces")
        with self.assertRaises(ConversationError):
            self.store.remember("one", "bad key", "x", source="test")


if __name__ == "__main__":
    unittest.main()
