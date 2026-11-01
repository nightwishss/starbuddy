import tempfile
import threading
import time
import unittest
from pathlib import Path

from event_service.store import EventError, EventStore


class EventStoreTests(unittest.TestCase):
    def test_publish_read_filter_and_privacy(self):
        store = EventStore(capacity=16, clock=lambda: 123.0)
        event = store.publish(topic="perception.posture.changed", source="pose",
                              payload={"state": "suspect"}, privacy="local-private")
        store.publish(topic="audio.output.completed", source="output", payload={"job_id": "j"})
        self.assertEqual(event["revision"], 1)
        value = store.read(topics=["perception.posture.changed"])
        self.assertEqual([x["topic"] for x in value["events"]], ["perception.posture.changed"])
        self.assertEqual(value["events"][0]["privacy"], "local-private")

    def test_ring_reports_gap_and_never_accepts_raw_blob(self):
        store = EventStore(capacity=16)
        for index in range(20):
            store.publish(topic="system.sample", source="test", payload={"n": index})
        value = store.read(after_revision=1)
        self.assertTrue(value["gap"])
        with self.assertRaises(EventError):
            store.publish(topic="camera.frame", source="test", payload={"raw": "x" * 20000})

    def test_wait_wakes_for_matching_topic(self):
        store = EventStore(capacity=16)
        result = []
        thread = threading.Thread(target=lambda: result.append(
            store.wait(after_revision=0, topics=["system.ready"], timeout=1)))
        thread.start()
        time.sleep(0.02)
        store.publish(topic="system.ready", source="test", payload={"ready": True})
        thread.join(1)
        self.assertEqual(result[0]["events"][0]["payload"], {"ready": True})


if __name__ == "__main__":
    unittest.main()
