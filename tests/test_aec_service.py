#!/usr/bin/env python3

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from audio_service.echo_cancel import (
    AEC_SINK_NAME, AEC_SOURCE_NAME, EchoCancelError, EchoCancelService,
)


class FakeManager:
    def __init__(self, inventories):
        self.inventories = list(inventories)
        self.last_inventory = self.inventories[-1] if self.inventories else []

    def inventory(self):
        if self.inventories:
            self.last_inventory = self.inventories.pop(0)
        return self.last_inventory

    def resolve(self, role):
        if role == "audio.primary_input":
            return {"device": {"device_id": "audio_input:usb", "endpoint": "physical.input"}}
        if role == "audio.primary_output":
            return {"device": {"device_id": "audio_output:usb", "endpoint": "physical.output"}}
        raise AssertionError(role)


VIRTUALS = [
    {"kind": "audio_input", "stable_id": AEC_SOURCE_NAME},
    {"kind": "audio_output", "stable_id": AEC_SINK_NAME},
]


class EchoCancelServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state = Path(self.temp.name) / "state.json"

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def command(arguments, **_kwargs):
        output = "17\n" if arguments[1] == "load-module" else ""
        return subprocess.CompletedProcess(arguments, 0, output, "")

    def test_owns_virtual_graph_and_writes_non_endpoint_state(self):
        service = EchoCancelService(
            manager=FakeManager([[], VIRTUALS]), command=self.command,
            state_path=self.state, sleep=MagicMock(),
        )

        started = service.start()
        self.assertEqual(started["status"], "ready")
        self.assertEqual(service.module_id, "17")
        self.assertNotIn("endpoint", self.state.read_text())
        stopped = service.stop()
        self.assertEqual(stopped["status"], "inactive")
        self.assertIsNone(service.module_id)
        self.assertEqual(json.loads(self.state.read_text())["module_loaded"], False)

    def test_refuses_preexisting_unowned_virtual_node(self):
        service = EchoCancelService(
            manager=FakeManager([VIRTUALS]), command=self.command,
            state_path=self.state,
        )
        with self.assertRaisesRegex(EchoCancelError, "without service ownership"):
            service.start()

    def test_missing_virtual_nodes_unloads_after_bounded_wait(self):
        command = MagicMock(side_effect=self.command)
        service = EchoCancelService(
            manager=FakeManager([[]]), command=command,
            state_path=self.state, sleep=MagicMock(),
        )
        with self.assertRaisesRegex(EchoCancelError, "did not become ready"):
            service.start()
        self.assertIsNone(service.module_id)
        self.assertTrue(any(call.args[0][1] == "unload-module" for call in command.call_args_list))


if __name__ == "__main__":
    unittest.main()
