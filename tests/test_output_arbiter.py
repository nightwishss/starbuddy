import time
import unittest

from output_service.arbiter import OutputArbiter, OutputError


class FakeEngine:
    def __init__(self):
        self.started = []
        self.stopped = []
        self.released = []
        self.states = {}

    def synthesize(self, text, owner):
        return {"media_id": "generated_ABCDEFGHIJKLMNOPQRSTUVWXYZ"}

    def start_playback(self, media_id, owner, volume_percent=70):
        playback = "playback_ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.started.append((owner, volume_percent))
        self.states[playback] = "playing"
        return {"playback_id": playback}

    def playback_status(self, playback_id):
        return {"state": self.states[playback_id]}

    def stop_playback(self, playback_id):
        self.stopped.append(playback_id)
        self.states[playback_id] = "stopped"

    def release(self, media_id, owner):
        self.released.append((media_id, owner))


class OutputArbiterTests(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.arbiter = OutputArbiter(self.engine)
        self.arbiter.start()

    def tearDown(self):
        self.arbiter.close()

    def wait_state(self, job_id, expected, timeout=1):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = self.arbiter.status(job_id=job_id)
            if value["state"] in expected:
                return value
            time.sleep(0.01)
        self.fail(f"state did not reach {expected}")

    def test_notification_never_interrupts_conversation(self):
        conversation = self.arbiter.say(text="hello", owner="session:test", category="conversation")
        self.wait_state(conversation["job_id"], {"playing"})
        note = self.arbiter.say(text="posture", owner="session:posture", category="notification",
                                expires_in_seconds=5)
        time.sleep(0.05)
        self.assertEqual(self.arbiter.status(job_id=note["job_id"])["state"], "queued")
        self.assertEqual(self.engine.stopped, [])

    def test_conversation_preempts_notification(self):
        note = self.arbiter.say(text="posture", owner="session:posture", category="notification")
        self.wait_state(note["job_id"], {"playing"})
        conversation = self.arbiter.say(text="answer", owner="session:test", category="conversation")
        self.wait_state(note["job_id"], {"interrupted"})
        self.wait_state(conversation["job_id"], {"playing"})
        self.assertTrue(self.engine.stopped)

    def test_validation_and_cancel_queued(self):
        first = self.arbiter.say(text="hello", owner="session:test")
        self.wait_state(first["job_id"], {"playing"})
        second = self.arbiter.say(text="later", owner="session:test", category="notification")
        self.assertEqual(self.arbiter.cancel(job_id=second["job_id"])["state"], "cancelled")
        with self.assertRaises(OutputError):
            self.arbiter.say(text="x", owner="bad owner")

    def test_caller_owned_media_is_not_released(self):
        value = self.arbiter.play_media(
            media_id="generated_ABCDEFGHIJKLMNOPQRSTUVWXYZ", owner="session:test")
        self.wait_state(value["job_id"], {"playing"})
        self.engine.states["playback_ABCDEFGHIJKLMNOPQRSTUVWXYZ"] = "completed"
        self.wait_state(value["job_id"], {"completed"})
        self.assertEqual(self.engine.released, [])


if __name__ == "__main__":
    unittest.main()
