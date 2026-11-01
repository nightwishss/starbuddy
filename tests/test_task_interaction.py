import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from event_service.store import EventStore
from task_interaction_service import TaskEventConsumer


def event(topic="task.notification.requested", revision=1, task_id="task_test"):
    return {
        "revision": revision, "topic": topic, "source": "task-orchestrator",
        "trace_id": task_id,
        "payload": {
            "task_id": task_id, "state": "completed", "summary": "喝水提醒",
            "notification": "该喝水了", "run_at": 1000.0, "step": None,
        },
    }


class FakeEvents:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []

    def wait(self, revision, **kwargs):
        self.calls.append((revision, kwargs))
        return self.results.pop(0) if self.results else {
            "revision": revision, "oldest_revision": revision + 1, "gap": False, "events": [],
        }


class FakeDisplay:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def update(self, **changes):
        self.calls.append(changes)
        if self.fail:
            raise RuntimeError("display unavailable")
        return changes


class FakeOutput:
    def __init__(self, fail=False, state="completed"):
        self.fail = fail
        self.final_state = state
        self.calls = []

    def say(self, text, owner, **kwargs):
        self.calls.append((text, owner, kwargs))
        if self.fail:
            raise RuntimeError("output unavailable")
        return {"job_id": "job_test", "state": "queued"}

    def status(self, job_id):
        return {"job_id": job_id, "state": self.final_state}


class TaskInteractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "consumer.json"

    def tearDown(self):
        self.temp.cleanup()

    def consumer(self, *, events=None, display=None, output=None):
        return TaskEventConsumer(
            events or FakeEvents(), display or FakeDisplay(), output or FakeOutput(),
            self.path, sleep=lambda _seconds: None, output_poll_seconds=0,
        )

    def test_notification_updates_display_uses_arbiter_and_returns_idle(self):
        display, output = FakeDisplay(), FakeOutput()
        consumer = self.consumer(display=display, output=output)
        consumer.handle(event())
        self.assertEqual([call["phase"] for call in display.calls],
                         ["attention", "success", "idle"])
        self.assertEqual(len(output.calls), 1)
        self.assertEqual(output.calls[0][1], "task:task_test")
        self.assertEqual(output.calls[0][2]["category"], "alert")
        self.assertEqual(output.calls[0][2]["trace_id"], "task_test")
        self.assertTrue(all(call["producer"] == "task" for call in display.calls))

    def test_task_visual_speaking_waits_for_actual_playback(self):
        class QueuedOutput(FakeOutput):
            def __init__(self):
                super().__init__()
                self.states = iter(["queued", "playing", "completed"])
            def status(self, job_id):
                return {"job_id": job_id, "state": next(self.states)}
        display = FakeDisplay()
        self.consumer(display=display, output=QueuedOutput()).handle(event())
        self.assertEqual([call["phase"] for call in display.calls],
                         ["attention", "speaking", "success", "idle"])

    def test_duplicate_event_and_restart_do_not_repeat_tts(self):
        output = FakeOutput()
        consumer = self.consumer(output=output)
        consumer.handle(event())
        consumer.handle(event(revision=2))
        self.assertEqual(len(output.calls), 1)
        restarted = self.consumer(output=output)
        restarted.handle(event(revision=3))
        self.assertEqual(len(output.calls), 1)

    def test_output_unavailable_keeps_display_and_returns_idle(self):
        display = FakeDisplay()
        consumer = self.consumer(display=display, output=FakeOutput(fail=True))
        consumer.handle(event())
        self.assertEqual(display.calls[0]["phase"], "attention")
        self.assertEqual(display.calls[-1]["phase"], "idle")
        self.assertEqual(consumer.state["deliveries"]["task.notification.requested:task_test"]["state"],
                         "failed")

    def test_display_unavailable_does_not_block_tts(self):
        output = FakeOutput()
        consumer = self.consumer(display=FakeDisplay(fail=True), output=output)
        consumer.handle(event())
        self.assertEqual(len(output.calls), 1)

    def test_skill_lifecycle_uses_skill_specific_display_copy(self):
        display=FakeDisplay(); consumer=self.consumer(display=display)
        value=event(topic="task.started")
        value["payload"]["task_kind"]="skill"
        value["payload"]["summary"]="持续 30 分钟坐姿守护"
        consumer.handle(value)
        self.assertEqual(display.calls[-1]["primary_text"],"正在启动坐姿守护")

    def test_both_unavailable_do_not_poison_later_event(self):
        output = FakeOutput(fail=True)
        consumer = self.consumer(display=FakeDisplay(fail=True), output=output)
        consumer.handle(event())
        consumer.handle(event(task_id="task_next", revision=2))
        self.assertEqual(len(output.calls), 2)  # ambiguous failures are never replayed

    def test_output_timeout_is_not_retried_to_avoid_duplicate_speech(self):
        class TimeoutOutput(FakeOutput):
            def say(self, text, owner, **kwargs):
                self.calls.append((text, owner, kwargs))
                raise TimeoutError("ambiguous admission timeout")
        output = TimeoutOutput()
        consumer = self.consumer(output=output)
        consumer.handle(event())
        self.assertEqual(len(output.calls), 1)

    def test_terminal_output_failure_is_recorded_and_not_replayed(self):
        output = FakeOutput(state="interrupted")
        consumer = self.consumer(output=output)
        consumer.handle(event())
        record = consumer.state["deliveries"]["task.notification.requested:task_test"]
        self.assertEqual(record["state"], "failed")
        self.assertEqual(record["output_state"], "interrupted")
        consumer.handle(event(revision=2))
        self.assertEqual(len(output.calls), 1)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_save_after_output_admission_does_not_get_swallowed(self):
        consumer = self.consumer()
        save = consumer._save
        calls = []
        def fail_after_admission():
            calls.append(1)
            if len(calls) == 2:
                raise OSError("disk full")
            save()
        with patch.object(consumer, "_save", side_effect=fail_after_admission):
            with self.assertRaises(OSError):
                consumer.handle(event())
        self.assertEqual(json.loads(self.path.read_text())["deliveries"][
            "task.notification.requested:task_test"]["state"], "attempting")

    def test_gap_without_retained_events_advances_cursor_not_busy_loop(self):
        # Regression: when the ring buffer has dropped everything behind our
        # cursor AND no retained event matches our topics, wait() returns
        # immediately with gap=True and events=[]. The consumer MUST advance the
        # cursor past the gap, otherwise run_forever() busy-loops (100% CPU).
        events = FakeEvents([{
            "revision": 6372, "oldest_revision": 6372, "gap": True, "events": [],
        }])
        consumer = self.consumer(events=events)
        result = consumer.poll_once(timeout=0)
        self.assertTrue(result["gap"])
        self.assertEqual(consumer.state["revision"], 6372)  # jumped to latest, gap skipped
        # A second poll must NOT re-report a gap at the same cursor.
        events.results.append({
            "revision": 6372, "oldest_revision": 6372, "gap": False, "events": [],
        })
        result2 = consumer.poll_once(timeout=0)
        self.assertFalse(result2["gap"])

    def test_gap_processes_retained_events_and_persists_cursor(self):
        events = FakeEvents([{
            "revision": 8, "oldest_revision": 8, "gap": True,
            "events": [event(revision=8)],
        }])
        output = FakeOutput()
        consumer = self.consumer(events=events, output=output)
        result = consumer.poll_once(timeout=0)
        self.assertTrue(result["gap"])
        self.assertEqual(consumer.state["revision"], 8)
        self.assertEqual(len(output.calls), 1)
        self.assertEqual(json.loads(self.path.read_text())["revision"], 8)

    def test_gap_with_multiple_pages_does_not_skip_retained_notifications(self):
        store = EventStore(capacity=128)
        for index in range(200):
            value = event(task_id=f"task_{index}")
            store.publish(topic=value["topic"], source="test", payload=value["payload"])
        class Client:
            def wait(self, revision, **kwargs):
                return store.wait(after_revision=revision, **kwargs)
        output = FakeOutput()
        consumer = self.consumer(events=Client(), output=output)
        consumer.state["revision"] = 1
        consumer.poll_once(timeout=0)
        self.assertEqual(consumer.state["revision"], 136)
        consumer.poll_once(timeout=0)
        self.assertEqual(consumer.state["revision"], 200)
        self.assertEqual(len(output.calls), 128)

    def test_handler_storage_error_does_not_acknowledge_event(self):
        consumer = self.consumer(events=FakeEvents([{
            "revision": 10, "gap": True, "events": [event(revision=10)],
        }]))
        with patch.object(consumer, "handle", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                consumer.poll_once(timeout=0)
        self.assertEqual(consumer.state["revision"], 0)

    def test_empty_filtered_page_advances_without_gap(self):
        consumer = self.consumer(events=FakeEvents([{
            "revision": 400, "gap": False, "events": [],
        }]))
        consumer.poll_once(timeout=0)
        self.assertEqual(consumer.state["revision"], 400)

    def test_failed_checkpoint_does_not_advance_memory_cursor(self):
        consumer = self.consumer()
        with patch.object(consumer, "_save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                consumer._checkpoint(10)
        self.assertEqual(consumer.state["revision"], 0)

    def test_corrupt_dedup_state_fails_closed_without_overwrite(self):
        self.path.write_text('{"revision": broken')
        with self.assertRaises(ValueError):
            self.consumer()
        self.assertEqual(self.path.read_text(), '{"revision": broken')

    def test_failure_backoff_is_bounded_and_cancellable(self):
        consumer = self.consumer()
        delays = []
        def wait(delay):
            delays.append(delay)
            if len(delays) == 8:
                consumer.stop()
        with patch.object(consumer, "poll_once", side_effect=OSError("unavailable")):
            with patch.object(consumer.stop_event, "wait", side_effect=wait):
                consumer.run_forever()
        self.assertEqual(delays, [2, 4, 8, 16, 30, 30, 30, 30])

    def test_event_bus_restart_resets_only_cursor_not_dedup(self):
        first = self.consumer(output=FakeOutput())
        first.handle(event())
        first.state["revision"] = 9
        first._save()
        events = FakeEvents([{
            "revision": 0, "oldest_revision": 1, "gap": False, "events": [],
        }])
        restarted = self.consumer(events=events, output=FakeOutput())
        restarted.poll_once(timeout=0)
        self.assertEqual(restarted.state["revision"], 0)
        self.assertIn("task.notification.requested:task_test", restarted.state["deliveries"])

    def test_malformed_event_advances_cursor_and_next_event_runs(self):
        bad = event(revision=1)
        bad["payload"]["notification"] = ""
        events = FakeEvents([{
            "revision": 2, "oldest_revision": 1, "gap": False,
            "events": [bad, event(revision=2, task_id="task_next")],
        }])
        output = FakeOutput()
        consumer = self.consumer(events=events, output=output)
        consumer.poll_once(timeout=0)
        self.assertEqual(consumer.state["revision"], 2)
        self.assertEqual(len(output.calls), 1)


if __name__ == "__main__":
    unittest.main()
