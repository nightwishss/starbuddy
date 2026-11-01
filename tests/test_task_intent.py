import tempfile
import threading
import unittest
from pathlib import Path

from gateway.display.server import DisplayState, KioskInteraction
from task_orchestrator.intent import IntentError, ReminderIntentRouter, TaskDraftStore


class FakeLLM:
    def chat(self, *_args, **_kwargs):
        return "普通聊天回答"


class FakeSpeech:
    pass


class FakeAudio:
    def __init__(self):
        self.played = []

    def start_playback(self, media_id, owner, **kwargs):
        self.played.append((media_id, owner, kwargs))
        return {"playback_id": f"playback_{len(self.played)}"}

    def playback_status(self, playback_id):
        return {"playback_id": playback_id, "state": "completed"}

    def stop_playback(self, _playback_id):
        return {"state": "stopped"}


class FakeOutput:
    def __init__(self):
        self.jobs = []

    def say(self, text, owner, **kwargs):
        job = {"job_id": f"job_{len(self.jobs)}", "state": "queued"}
        self.jobs.append((text, owner, kwargs, job))
        return job

    def status(self, job_id):
        return {"job_id": job_id, "state": "completed"}

    def cancel(self, _job_id):
        return {"state": "cancelled"}


class FakeTaskClient:
    def __init__(self):
        self.calls = []

    def create(self, utterance, **kwargs):
        self.calls.append((utterance, kwargs))
        return {"task_id": "task_test", "state": "scheduled", "summary": "一分钟后提醒：该喝水了"}


class TaskIntentTests(unittest.TestCase):
    def setUp(self):
        self.router = ReminderIntentRouter()

    def test_normal_chat_is_not_a_reminder(self):
        for text in ("你觉得今天怎么样", "喝水很重要吗", "我刚才喝了水"):
            self.assertEqual(self.router.classify(text)["intent"], "chat")

    def test_reminder_builds_bounded_draft_without_creating_task(self):
        result = self.router.classify(
            "1分钟后提醒我喝水", timezone="Europe/Madrid",
            source_conversation_id="session.a",
        )
        self.assertEqual(result["intent"], "task")
        self.assertEqual(result["draft"]["delay_seconds"], 60)
        self.assertEqual(result["draft"]["steps"], [])
        self.assertEqual(result["draft"]["notification"], "该喝水了")

    def test_ambiguous_task_request_is_not_chat(self):
        result = self.router.classify("提醒我喝水")
        self.assertEqual(result["intent"], "ambiguous")

    def test_posture_duration_builds_skill_draft(self):
        result=self.router.classify("接下来半小时帮我看着坐姿，坐不好提醒我")
        self.assertEqual(result["intent"],"task")
        self.assertEqual(result["draft"]["schedule"],{"kind":"now"})
        self.assertEqual(result["draft"]["steps"][0]["skill"],"monitor_posture")
        self.assertEqual(result["draft"]["steps"][0]["arguments"]["duration_seconds"],1800)
        work=self.router.classify("我要工作30分钟，如果坐姿不好就提醒我")
        self.assertEqual(work["draft"]["steps"][0]["arguments"]["duration_seconds"],1800)

    def test_explicit_matrix_prelude_composes_with_posture_skill(self):
        result = self.router.classify("先让矩阵显示爱心，然后监测坐姿十分钟")
        self.assertEqual(result["intent"], "task")
        self.assertEqual([next(iter(step)) for step in result["draft"]["steps"]],
                         ["capability", "skill"])
        self.assertEqual(result["draft"]["steps"][0]["arguments"], {"name": "heart"})
        self.assertEqual(result["draft"]["steps"][1]["arguments"], {"duration_seconds": 600})
        ambiguous = self.router.classify("让灯和坐姿监测一起运行十分钟")
        self.assertEqual(ambiguous["intent"], "ambiguous")
        reversed_order = self.router.classify("先监测坐姿十分钟，然后让矩阵显示爱心")
        self.assertEqual(reversed_order["intent"], "ambiguous")

    def test_draft_ttl_session_isolation_and_superseding(self):
        now = [100.0]
        store = TaskDraftStore(clock=lambda: now[0], ttl_seconds=10)
        first = store.create({"source_conversation_id": "session.a", "summary": "a"})
        with self.assertRaises(IntentError):
            store.get(first["draft_id"], source_conversation_id="session.b")
        second = store.create({"source_conversation_id": "session.a", "summary": "b"})
        with self.assertRaises(IntentError):
            store.get(first["draft_id"], source_conversation_id="session.a")
        now[0] = 111.0
        with self.assertRaises(IntentError):
            store.get(second["draft_id"], source_conversation_id="session.a")

    def test_confirm_creates_once_and_duplicate_confirm_is_rejected(self):
        tasks = FakeTaskClient()
        interaction = KioskInteraction(
            DisplayState(), FakeLLM(), FakeSpeech(), FakeAudio(),
            output=FakeOutput(), task_client=tasks,
        )
        result = interaction.converse("1分钟后提醒我喝水")
        self.assertEqual(tasks.calls, [])
        draft_id = result["task_draft"]["draft_id"]
        confirmed = interaction.confirm_task(draft_id)
        self.assertTrue(confirmed["confirmed"])
        self.assertEqual(len(tasks.calls), 1)
        self.assertEqual(tasks.calls[0][1]["source_conversation_id"], "starbuddy.hdmi")
        with self.assertRaises(IntentError):
            interaction.confirm_task(draft_id)
        self.assertEqual(len(tasks.calls), 1)

    def test_cancel_never_creates_task(self):
        tasks = FakeTaskClient()
        interaction = KioskInteraction(
            DisplayState(), FakeLLM(), FakeSpeech(), FakeAudio(),
            output=FakeOutput(), task_client=tasks,
        )
        result = interaction.converse("一分钟后提醒我喝水")
        cancelled = interaction.cancel_task_draft(result["task_draft"]["draft_id"])
        self.assertTrue(cancelled["cancelled"])
        self.assertEqual(tasks.calls, [])

    def test_posture_skill_confirmation_preserves_now_schedule_and_skill_step(self):
        tasks=FakeTaskClient()
        interaction=KioskInteraction(
            DisplayState(),FakeLLM(),FakeSpeech(),FakeAudio(),output=FakeOutput(),task_client=tasks,
        )
        result=interaction.converse("接下来半小时帮我看着坐姿，坐不好提醒我")
        interaction.confirm_task(result["task_draft"]["draft_id"])
        proposal=tasks.calls[0][1]["proposal"]
        self.assertEqual(proposal["schedule"],{"kind":"now"})
        self.assertEqual(proposal["steps"][0]["skill"],"monitor_posture")

    def test_concurrent_duplicate_confirm_creates_exactly_once(self):
        entered, release = threading.Event(), threading.Event()

        class BlockingTaskClient(FakeTaskClient):
            def create(self, utterance, **kwargs):
                self.calls.append((utterance, kwargs))
                entered.set()
                release.wait(2)
                return {"task_id": "task_test", "state": "scheduled", "summary": "提醒"}

        tasks = BlockingTaskClient()
        interaction = KioskInteraction(
            DisplayState(), FakeLLM(), FakeSpeech(), FakeAudio(),
            output=FakeOutput(), task_client=tasks,
        )
        draft_id = interaction.converse("1分钟后提醒我喝水")["task_draft"]["draft_id"]
        results = []
        thread = threading.Thread(target=lambda: results.append(interaction.confirm_task(draft_id)))
        thread.start()
        self.assertTrue(entered.wait(1))
        with self.assertRaisesRegex(IntentError, "already in progress"):
            interaction.confirm_task(draft_id)
        release.set()
        thread.join(2)
        self.assertEqual(len(tasks.calls), 1)
        self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
