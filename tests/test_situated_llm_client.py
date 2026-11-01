"""Pure transport-spy checks; no model, devices or production sockets."""
import json
from contextlib import contextmanager
import unittest
from unittest.mock import patch

from gateway.llm import LocalLLMClient, LocalLLMError
from gateway.llm.client import MAX_PROPOSAL_PROMPT_BYTES, SITUATED_SYSTEM_PROMPT


class SituatedLLMClientTests(unittest.TestCase):
    def test_situated_route_has_own_budget_and_no_ordinary_chat_cleanup(self):
        raw = '{"intent":"chat","response":"你好","requested_capability":null}'
        client = LocalLLMClient()
        with patch.object(client, '_completion', return_value=raw) as complete:
            self.assertEqual(client.propose_situated('合成宿主情境'), raw)
        payload = complete.call_args.args[0]
        self.assertEqual(payload['max_completion_tokens'], 256)
        self.assertIs(payload['stream'], False)
        self.assertEqual(payload['temperature'], .1)
        self.assertEqual(payload['messages'][0]['content'], SITUATED_SYSTEM_PROMPT)
        self.assertEqual(payload['messages'][1]['content'], '合成宿主情境')
        self.assertNotIn('ordinary_chat', complete.call_args.kwargs)

    def test_legacy_proposal_budget_unchanged(self):
        client = LocalLLMClient()
        with patch.object(client, '_completion', return_value='{}') as complete:
            client.propose('静态能力目录')
        self.assertEqual(complete.call_args.args[0]['max_completion_tokens'], 512)

    def test_situated_prompt_bounds_apply_before_transport(self):
        client = LocalLLMClient()
        with patch.object(client, '_completion') as complete:
            for prompt in ('', '   ', None, '中' * (MAX_PROPOSAL_PROMPT_BYTES // 3 + 1)):
                with self.subTest(prompt_type=type(prompt).__name__):
                    with self.assertRaises(ValueError):
                        client.propose_situated(prompt)
            complete.assert_not_called()

    def test_situated_uses_shared_completion_to_post_path_once(self):
        raw = json.dumps({'choices': [{'message': {'content': '{}'}}]}).encode()
        client = LocalLLMClient()
        with patch.object(client, '_post', return_value=raw) as post:
            self.assertEqual(client.propose_situated('合成情境'), '{}')
        post.assert_called_once()

    def test_foreground_and_background_use_same_lease_terminal_ack(self):
        raw = json.dumps({'choices': [{'message': {'content': '{}'}}]}).encode()
        for kind in ('foreground', 'background'):
            events = []
            class Permit:
                terminal = False
                def acknowledge_terminal(self):
                    self.terminal = True
            @contextmanager
            def admission(**values):
                permit = Permit()
                events.append(values)
                try:
                    yield permit
                finally:
                    events.append(permit.terminal)
            client = LocalLLMClient(timeout=15, admission_kind=kind)
            with patch('gateway.llm.client.local_llm_admission', admission), patch.object(client, '_post_native', return_value=raw):
                self.assertEqual(client.propose_situated('合成情境'), '{}')
            self.assertEqual(events, [{'owner': 'local-llm-client', 'kind': kind, 'timeout': 15}, True])

    def test_incomplete_native_call_never_acknowledges_terminal(self):
        permits = []
        class Permit:
            terminal = False
            def acknowledge_terminal(self):
                self.terminal = True
        @contextmanager
        def admission(**values):
            permit = Permit()
            permits.append(permit)
            yield permit
        client = LocalLLMClient(admission_kind='background')
        with patch('gateway.llm.client.local_llm_admission', admission), patch.object(client, '_post_native', side_effect=TimeoutError):
            with self.assertRaises(LocalLLMError):
                client.propose_situated('合成情境')
        self.assertEqual(len(permits), 1)
        self.assertFalse(permits[0].terminal)

    def test_invalid_admission_kinds_rejected(self):
        for kind in (None, True, {}, 'urgent', 'BACKGROUND'):
            with self.assertRaises(ValueError):
                LocalLLMClient(admission_kind=kind)

    def test_transport_failure_is_not_retried_and_enters_cooldown(self):
        client = LocalLLMClient()
        with patch.object(client, '_post', side_effect=TimeoutError) as post:
            with self.assertRaises(LocalLLMError):
                client.propose_situated('合成情境')
            with self.assertRaises(LocalLLMError):
                client.propose_situated('合成情境')
        post.assert_called_once()

    def test_oversize_answer_rejected(self):
        raw = json.dumps({'choices': [{'message': {'content': 'x' * 8193}}]}).encode()
        client = LocalLLMClient()
        with patch.object(client, '_post', return_value=raw):
            with self.assertRaises(LocalLLMError):
                client.propose_situated('合成情境')
