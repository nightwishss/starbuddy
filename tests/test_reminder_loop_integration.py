import json
import tempfile
import unittest
from pathlib import Path

from gateway.display.server import DisplayState, KioskInteraction
from task_interaction_service import TaskEventConsumer
from task_orchestrator.engine import TaskOrchestrator
from task_orchestrator.planner import TaskPlanner
from task_orchestrator.store import TaskStore


class FakeModel:
    def chat(self, _prompt, **_options):
        return json.dumps({
            "summary": "一分钟后提醒喝水",
            "schedule": {"kind": "delay", "delay_seconds": 60},
            "notification": "该喝水了", "steps": [],
        }, ensure_ascii=False)


class FakeRuntime:
    def call(self, *_args, **_kwargs):
        raise AssertionError("pure reminder must not invoke Runtime")


class CapturingEvents:
    def __init__(self):
        self.events = []

    def publish(self, topic, source, payload, **options):
        value = {
            "revision": len(self.events) + 1, "topic": topic, "source": source,
            "payload": payload, **options,
        }
        self.events.append(value)
        return value


class OrchestratorClient:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.calls = 0

    def create(self, utterance, **kwargs):
        self.calls += 1
        return self.orchestrator.create(utterance, **kwargs)


class FakeLLM:
    def chat(self, *_args, **_kwargs):
        return "unused"


class FakeSpeech:
    pass


class FakeAudio:
    def start_playback(self, *_args, **_kwargs):
        return {"playback_id": "playback_confirmation"}

    def playback_status(self, playback_id):
        return {"playback_id": playback_id, "state": "completed"}

    def stop_playback(self, _playback_id):
        return {"state": "stopped"}


class FakeOutput:
    def __init__(self):
        self.calls = []

    def say(self, text, owner, **kwargs):
        self.calls.append((text, owner, kwargs))
        return {"job_id": f"job_{len(self.calls)}", "state": "queued"}

    def status(self, job_id):
        return {"job_id": job_id, "state": "completed"}

    def cancel(self, _job_id):
        return {"state": "cancelled"}


class FakeDisplay:
    def __init__(self):
        self.calls = []

    def update(self, **changes):
        self.calls.append(changes)
        return changes


class ReminderLoopIntegrationTests(unittest.TestCase):
    def test_one_minute_reminder_closes_the_embodied_loop(self):
        with tempfile.TemporaryDirectory() as temporary:
            now = [1000.0]
            clock = lambda: now[0]
            events = CapturingEvents()
            orchestrator = TaskOrchestrator(
                TaskPlanner(FakeModel(), clock=clock),
                TaskStore(Path(temporary) / "tasks.json", clock=clock),
                FakeRuntime(), events, clock=clock,
            )
            client = OrchestratorClient(orchestrator)
            confirmation_output = FakeOutput()
            interaction = KioskInteraction(
                DisplayState(), FakeLLM(), FakeSpeech(), FakeAudio(),
                output=confirmation_output, task_client=client,
            )

            proposal = interaction.converse("1分钟后提醒我喝水")
            self.assertEqual(client.calls, 0)
            self.assertTrue(proposal["confirmation_required"])
            confirmed = interaction.confirm_task(proposal["task_draft"]["draft_id"])
            self.assertEqual(client.calls, 1)
            task_id = confirmed["task"]["task_id"]
            self.assertEqual(orchestrator.get(task_id)["state"], "scheduled")
            self.assertTrue((Path(temporary) / "tasks.json").exists())

            # A fresh store sees the scheduled task: it survives an orchestrator restart.
            self.assertEqual(
                TaskStore(Path(temporary) / "tasks.json", clock=clock).get(task_id)["state"],
                "scheduled",
            )
            now[0] += 60
            orchestrator.run_due(now=now[0])
            notification = next(item for item in events.events
                                if item["topic"] == "task.notification.requested")
            display, output = FakeDisplay(), FakeOutput()
            consumer = TaskEventConsumer(
                None, display, output, Path(temporary) / "task-interaction.json",
                sleep=lambda _seconds: None, output_poll_seconds=0,
            )
            consumer.handle(notification)

            self.assertEqual(orchestrator.get(task_id)["state"], "completed")
            self.assertEqual(len(output.calls), 1)
            self.assertEqual(output.calls[0][0], "该喝水了")
            self.assertEqual(display.calls[0]["phase"], "attention")
            self.assertEqual(display.calls[-1]["phase"], "idle")


if __name__ == "__main__":
    unittest.main()
