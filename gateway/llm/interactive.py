"""Latency-aware text replies; never accepts a perception packet or device tools.

Only the ordinary-dialogue projection or an explicitly public game prompt may
reach the optional API. Foreground dialogue can explicitly opt into API-first
replies; the default remains adaptive local-first. Native work is never abandoned
or raced against a second native call. Offline-only, per-turn local-only and the
existing credential gate remain authoritative.
"""
from __future__ import annotations

import json
import threading
import time

from governance.privacy import starbuddy_offline_only
from gateway.llm.dialogue_intent import DIALOGUE_CONTEXT_RULES


class InteractiveTextClient:
    # Lets the reasoner forward its existing cancellation/currentness callback
    # without changing direct local clients or older test/runtime adapters.
    supports_turn_current = True

    def __init__(self, local, cloud=None, *, clock=time.monotonic, slow_seconds=4,
                 local_ready=None, dialogue_cloud_first=False):
        self.local, self.cloud, self.clock = local, cloud, clock
        self.slow_seconds = slow_seconds
        self.local_ready = local_ready
        self.dialogue_cloud_first = dialogue_cloud_first is True
        self.prefer_api_until = 0.0
        self.last_route = 'local-bounded'
        self.last_fallback = False
        self.last_calls = 0
        self._lock = threading.Lock()

    def _cloud_ready(self):
        return (self.cloud is not None and not starbuddy_offline_only()
                and self.clock() >= getattr(self.cloud, '_retry_at', 0))

    def _local_ready(self):
        if self.clock() < getattr(self.local, '_retry_at', 0):
            return False
        inflight = getattr(self.local, '_inflight', None)
        if inflight is not None and inflight.locked():
            return False
        try:
            return self.local_ready() is True if self.local_ready is not None else True
        except Exception:
            return False

    def status(self):
        return {'cloud_fallback': self._cloud_ready(), 'last_route': self.last_route,
                'last_fallback': self.last_fallback,
                'policy': ('cloud-first-dialogue-text-only' if self.dialogue_cloud_first
                           else 'local-first-adaptive-text-only'),
                'prefer_api': self._cloud_ready() and (
                    self.dialogue_cloud_first or self.clock() < self.prefer_api_until)}

    def _check_current(self, current):
        if current is None:
            return
        try:
            valid = current() is True
        except Exception:
            valid = False
        if not valid:
            self.last_route = 'turn-stale'
            # A stopped turn is not a failed model answer or a reason to speak
            # a fixed fallback. The host retains authority over all delivery.
            raise RuntimeError('interactive turn is no longer current')

    def _run(self, local_call, cloud_call, fallback, *, prefer_api=False,
             allow_cloud=True, current=None):
        with self._lock:
            self.last_fallback = False
            self.last_calls = 0
            self._check_current(current)
            cloud_allowed = allow_cloud is True
            prefer_api = cloud_allowed and self._cloud_ready() and (
                prefer_api or self.clock() < self.prefer_api_until or not self._local_ready())
            routes = [('cloud-text-api', cloud_call), ('local-bounded', local_call)] if prefer_api else [
                ('local-bounded', local_call), ('cloud-text-api', cloud_call)]
            for route, call in routes:
                self._check_current(current)
                if route == 'cloud-text-api' and (not cloud_allowed or not self._cloud_ready()):
                    continue
                began = self.clock()
                try:
                    self.last_calls += 1
                    answer = call()
                    if not isinstance(answer, str) or not answer.strip():
                        raise ValueError('empty interactive reply')
                    if route == 'local-bounded' and self.clock() - began > self.slow_seconds:
                        self.prefer_api_until = self.clock() + 120
                except Exception:
                    if route == 'local-bounded':
                        self.prefer_api_until = self.clock() + 120
                    # The local synchronous call has returned. No background
                    # generation is launched, detached, cancelled or retried.
                    self._check_current(current)
                    continue
                self._check_current(current)
                self.last_route = route
                return answer.strip()
            self._check_current(current)
            self.last_route, self.last_fallback = 'fixed-last-resort', True
            return fallback

    def propose_dialogue(self, message, *, history, system_prompt, calculation=None,
                         allow_cloud=True, current=None):
        # The caller is SituatedReasoner's text-only projection, not the full
        # multimodal workspace. Keep complete, bounded RAM dialogue pairs only.
        if DIALOGUE_CONTEXT_RULES not in system_prompt:
            system_prompt += DIALOGUE_CONTEXT_RULES
        pairs, pending = [], None
        for row in history[-6:]:
            if row.get('role') == 'user':
                pending = str(row.get('text', ''))[:500]
            elif row.get('role') == 'assistant' and pending is not None:
                pairs += [{'role': 'user', 'content': pending},
                          {'role': 'assistant', 'content': str(row.get('text', ''))[:240]}]
                pending = None
        cloud_text = message
        if calculation is not None:
            cloud_text += '\n本轮计算依据（最终结果，不要再次运算）：' + json.dumps(calculation, ensure_ascii=False)
        # Do not silently crop a question; a long turn simply remains local.
        def api():
            if len(cloud_text) > 500:
                raise ValueError('text too long for API boundary')
            return self.cloud.chat(cloud_text, history=pairs[-4:], purpose='context_dialogue')
        return self._run(
            lambda: self.local.propose_dialogue(message, history=history,
                system_prompt=system_prompt, calculation=calculation), api,
            '抱歉，这次没能生成回答。我们换个说法再试一次，好吗？',
            prefer_api=self.dialogue_cloud_first, allow_cloud=allow_cloud, current=current)

    def public_reply(self, prompt, *, api_prompt=None, dialogue=None, prefer_api=False, api_purpose='conversation', max_completion_tokens=128, temperature=.5, fallback):
        """Only authored, anonymous game instructions; not camera/profile data."""
        if api_purpose not in {'conversation', 'story'}:
            raise ValueError('unknown public text purpose')
        def api():
            if not isinstance(api_prompt, str) or not 0 < len(api_prompt) <= 500:
                raise ValueError('no approved API text')
            options = {'purpose': 'story'} if api_purpose == 'story' else {}
            return self.cloud.chat(api_prompt, **options)
        return self._run(
            lambda: self.local.compose_text(**dialogue, max_completion_tokens=max_completion_tokens,
                                    temperature=temperature) if dialogue is not None and hasattr(self.local,'compose_text') else
                    self.local.chat(prompt, max_completion_tokens=max_completion_tokens, temperature=temperature),
            api,
            fallback,prefer_api=prefer_api and api_prompt is not None)

    def propose_situated(self, prompt):
        self.last_route, self.last_fallback = 'local-bounded', False
        self.last_calls = 1
        return self.local.propose_situated(prompt)

