#!/usr/bin/env python3

import json
import unittest
from unittest.mock import MagicMock, patch

from gateway.llm import LocalLLMClient, LocalLLMError


class LocalLLMClientTests(unittest.TestCase):
    def test_rejects_non_loopback_endpoint(self):
        with self.assertRaises(ValueError):
            LocalLLMClient("http://192.168.1.2:9001")

    @patch("gateway.llm.client.urllib.request.urlopen")
    def test_returns_assistant_content(self, urlopen):
        response = MagicMock()
        response.read.return_value = json.dumps(
            {"choices": [{"message": {"content": " 板端回答 "}}]}
        ).encode()
        urlopen.return_value.__enter__.return_value = response

        answer = LocalLLMClient().chat(
            "你好", max_completion_tokens=48, temperature=0.2
        )

        self.assertEqual(answer, "板端回答")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "qwen3_4b_instruct_2507")
        self.assertEqual(payload["messages"][-1]["content"], "你好")
        self.assertEqual(payload["max_completion_tokens"], 48)
        self.assertEqual(payload["temperature"], 0.2)

    def test_rejects_invalid_generation_bounds(self):
        with self.assertRaises(ValueError):
            LocalLLMClient().chat("你好", max_completion_tokens=0)
        with self.assertRaises(ValueError):
            LocalLLMClient().chat("你好", temperature=3)

    @patch("gateway.llm.client.urllib.request.urlopen")
    def test_invalid_response_is_public_error(self, urlopen):
        response = MagicMock()
        response.read.return_value = b"{}"
        urlopen.return_value.__enter__.return_value = response
        with self.assertRaisesRegex(LocalLLMError, "格式无效"):
            LocalLLMClient().chat("你好")


if __name__ == "__main__":
    unittest.main()
