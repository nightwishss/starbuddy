import unittest

from conversation_service.pipeline import ConversationPipeline, TurnCancelled
from conversation_service.store import ConversationStore


class ConversationPipelineTests(unittest.TestCase):
    def test_adapter_neutral_turn_receives_history_and_memory(self):
        store = ConversationStore()
        store.remember("voice.one", "topic", "机器人", source="test")
        observed = []

        def respond(request):
            observed.append(request)
            return "我们继续聊机器人。"

        result = ConversationPipeline(
            store, respond, backend="test-local",
        ).run("voice.one", "继续刚才的话题", input_type="voice")
        self.assertEqual(result.answer, "我们继续聊机器人。")
        self.assertEqual(observed[0]["memory"]["topic"]["value"], "机器人")
        self.assertEqual(observed[0]["input_type"], "voice")
        self.assertEqual(store.context("voice.one")["messages"][-1]["text"], result.answer)

    def test_responder_failure_atomically_closes_turn(self):
        store = ConversationStore()

        def fail(_request):
            raise TimeoutError("model timeout")

        with self.assertRaises(TimeoutError):
            ConversationPipeline(store, fail, backend="test").run("api.one", "hello")
        context = store.context("api.one")
        self.assertIsNone(context["active_turn"])
        self.assertEqual(context["messages"], [])
        self.assertEqual(context["last_turn"]["status"], "failed")

    def test_responder_can_select_backend_per_turn(self):
        store = ConversationStore()
        result = ConversationPipeline(
            store, lambda _request: ("工具回答", "local-agent"),
        ).run("agent.one", "检查系统", input_type="event")
        self.assertEqual(result.backend, "local-agent")
        self.assertEqual(store.context("agent.one")["messages"][-1]["backend"], "local-agent")

    def test_interrupted_model_result_is_not_committed_to_session(self):
        store = ConversationStore()
        with self.assertRaises(TurnCancelled):
            ConversationPipeline(store, lambda _request: "late response", backend="local").run(
                "display.one", "hello", is_active=lambda: False,
            )
        context = store.context("display.one")
        self.assertEqual(context["messages"], [])
        self.assertEqual(context["last_turn"]["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
