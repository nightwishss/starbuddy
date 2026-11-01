import asyncio
import json
import os
import socket
import tempfile
import threading
import time
import unittest

from conversation_service.client import ConversationClient, ConversationClientError
from conversation_service.service import run


class ConversationServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.socket_path = os.path.join(self.temp.name, "conversation.sock")
        self.loop = asyncio.new_event_loop()
        self.task = None

        def serve():
            asyncio.set_event_loop(self.loop)
            self.task = self.loop.create_task(run(self.socket_path))
            try:
                self.loop.run_until_complete(self.task)
            except asyncio.CancelledError:
                pass
            finally:
                self.loop.run_until_complete(self.loop.shutdown_asyncgens())
                self.loop.close()

        self.thread = threading.Thread(target=serve, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 2
        while not os.path.exists(self.socket_path) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.client = ConversationClient(self.socket_path, timeout=2)

    def tearDown(self):
        if self.task is not None:
            self.loop.call_soon_threadsafe(self.task.cancel)
        self.thread.join(1)
        self.temp.cleanup()

    def test_round_trip_and_owner_only_socket(self):
        self.assertEqual(os.stat(self.socket_path).st_mode & 0o777, 0o600)
        opened = self.client.ensure("cli.test")
        turn = self.client.start_turn("cli.test", "你好", input_type="api")
        result = self.client.complete_turn(
            "cli.test", turn["active_turn"]["turn_id"], "你好呀", backend="local-qwen",
        )
        self.assertGreater(result["revision"], opened["revision"])
        self.assertEqual(result["messages"][-1]["backend"], "local-qwen")
        self.assertFalse(self.client.stats()["persistent"])

    def test_errors_have_stable_codes(self):
        self.client.start_turn("busy", "first")
        with self.assertRaises(ConversationClientError) as caught:
            self.client.start_turn("busy", "second")
        self.assertEqual(caught.exception.code, "turn_in_progress")

    def test_invalid_json_does_not_stop_service(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as raw:
            raw.connect(self.socket_path)
            raw.sendall(b"not-json\n")
            response = json.loads(raw.recv(4096))
        self.assertEqual(response["code"], "invalid_json")
        self.assertIn("active_sessions", self.client.stats())


if __name__ == "__main__":
    unittest.main()
