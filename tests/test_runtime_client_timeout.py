#!/usr/bin/env python3
"""Runtime client transport-budget regression tests."""

import unittest
from unittest.mock import MagicMock, patch

from runtime.client import RuntimeClient


class RuntimeClientTimeoutTests(unittest.TestCase):
    @staticmethod
    def _socket_with_response():
        sock = MagicMock()
        sock.__enter__.return_value = sock
        reader = MagicMock()
        reader.__enter__.return_value = reader
        reader.readline.return_value = b'{"ok":true,"result":{}}\n'
        sock.makefile.return_value = reader
        return sock

    @patch("runtime.client.socket.socket")
    def test_execution_budget_extends_transport_wait(self, socket_factory):
        sock = self._socket_with_response()
        socket_factory.return_value = sock

        RuntimeClient("/tmp/test.sock", timeout=10).call(
            "audio.capture_turn", timeout_ms=30_000,
        )

        sock.settimeout.assert_called_once_with(32.0)

    @patch("runtime.client.socket.socket")
    def test_short_execution_budget_keeps_configured_transport_wait(self, socket_factory):
        sock = self._socket_with_response()
        socket_factory.return_value = sock

        RuntimeClient("/tmp/test.sock", timeout=10).call(
            "test.echo", timeout_ms=1_000,
        )

        sock.settimeout.assert_called_once_with(10)


if __name__ == "__main__":
    unittest.main()

