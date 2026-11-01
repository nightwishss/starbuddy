"""Ephemeral, local-only host for the situated interaction product surface.

The model proposes; this host owns task truth, sensor permission and delivery.
No personal-memory or persisted conversation-history access occurs here.
"""
from __future__ import annotations

import math
import re
import threading
import time
from collections import Counter
from copy import deepcopy
from decimal import Decimal

from interaction_context import assemble_turn_context
from interaction_context.query import perception_query
from interaction_context.reasoning import build_reasoning_packet
from interaction_context.situation import SituationState, TASKS
from interaction_context.workspace import Workspace
from gateway.agent.situated import SituatedReasoner
from gateway.display.situated_audio import FixedResponseAudio
from gateway.display.situated_actions import ReminderActionBroker, ForegroundBudget
from gateway.display.arithmetic import ArithmeticContext


class SituatedInteraction:
    def __init__(self, kiosk, *, reasoner=None, situation=None, safety=None):
        self.kiosk = kiosk
        self.situation = situation or SituationState()
        self.reasoner = reasoner or SituatedReasoner(kiosk.local_llm)
        self.arithmetic = ArithmeticContext()
        self.safety = safety
        self.last_timing = {}
        self.last_face_context = None
        self.source_states = {}
        self.context_observed_at = None
        self.fixed_audio = FixedResponseAudio(getattr(kiosk, 'media', None), getattr(kiosk, 'output', None))
        self.workspace = Workspace(clock=lambda: time.time())
        task_client = getattr(kiosk, 'task_client', None)
        if task_client is not None and hasattr(task_client, 'socket_path'):
            from task_orchestrator.client import TaskClient
            task_client = TaskClient(task_client.socket_path, timeout=2.0)
        self.actions = ReminderActionBroker(task_client)
        self.budget = ForegroundBudget()
        self._context_lock = threading.RLock()
        self._action_control_lock = threading.RLock()
        self._control_revision = 0
        self._refresh_lock = threading.Lock()
        self._last_refresh = float('-inf')
        self._last_action_poll = float('-inf')
        self.delivery = {'state': 'idle', 'job_id': None, 'confirmed_terminal': False}
        self._delivery_turn = None

    def status(self, *, refresh=False):
        # Passive reads update anonymous metadata only, never ask a submodel to
        # inspect a frame. No background loop is created by this workspace.
        if refresh:
            self.refresh_context()
        value = self.situation.snapshot()
        value['tasks'] = [{key: row[key] for key in ('id', 'title', 'prompt')} for row in TASKS]
        interaction = self.kiosk.interaction_status()
        value['phase'] = interaction['phase']
        # Logical cancel revokes authority immediately, but native capture,
        # ASR, inference or Output may still be draining. Expose only actual
        # lock/native-readiness metadata; a missing probe is UNKNOWN, not off.
        voice_lock = getattr(self.kiosk, 'voice_lock', None)
        value['input_active'] = voice_lock.locked() if voice_lock is not None else None
        value['output_pending'] = interaction.get('output_pending')
        value['output_ack_unconfirmed'] = interaction.get('output_ack_unconfirmed')
        value['control_epoch'] = interaction.get('control_epoch')
        value['control_revision'] = interaction.get('control_revision')
        mode_status = getattr(self.kiosk, 'mode_status', None)
        try:
            readiness = mode_status() if callable(mode_status) else {}
            value['voice_ready'] = readiness.get('ready') is True
        except Exception:
            value['voice_ready'] = False
        value['source_states'] = dict(self.source_states)
        value['context_observed_at'] = self.context_observed_at
        value['context_is_live'] = False
        value['offers_enabled'] = value.get('offers', {}).get('enabled', False)
        if value.get('decision'):
            value['decision']['target'] = (value.get('task') or {}).get('title', '本轮交互')
        value['timing_ms'] = dict(self.last_timing)
        value['face_context'] = deepcopy(self.last_face_context)
        value['privacy'] = {'persistent': False, 'speaker_binding': 'unconfirmed',
                            'profile_read': False, 'automatic_capture': False}
        value['fixed_voice_feedback'] = self.fixed_audio.status()
        value['workspace'] = self.workspace.snapshot()
        value['action'] = self.actions.status()
        value['scheduler'] = self.budget.status()
        with self._context_lock:
            value['delivery'] = dict(self.delivery)
        value['privacy']['transient_dialogue'] = {'max_messages': 6, 'ttl_seconds': 900,
                                                  'storage': 'process_ram_only'}
        status = getattr(self.reasoner, 'status', None)
        backend = status() if callable(status) else {'backend': 'injected-test'}
        return {'ok': True, 'state': value, 'backend': backend}

    def _bump_control(self):
        with self._context_lock:
            self._control_revision += 1
            self.arithmetic.invalidate()
        self.budget.cancel()

    def _read_context(self, request=None):
        request = request or {'user_text': '', 'input_type': 'text',
                              'session_id': 'starbuddy.situated', 'messages': [], 'memory': {}}
        try:
            admitted = assemble_turn_context(request, self.kiosk._perception_context(), None, [])
            packet = build_reasoning_packet(request, admitted)
        except Exception:
            # A failed source read revokes previous attention; never preserve a
            # stale scene just because the producer cannot currently be read.
            packet = build_reasoning_packet(request, assemble_turn_context(request, None, None, []))
        self.source_states = {key: row['state'] for key, row in packet['source_states'].items()}
        self.context_observed_at = packet['basis'].get('captured_at')
        self.workspace.ingest(packet)
        return packet

    def refresh_context(self):
        now = time.monotonic()
        if now - self._last_refresh < 2 or not self._refresh_lock.acquire(blocking=False):
            return
        try:
            self._last_refresh = now
            self._read_context()
            if now - self._last_action_poll >= 3:
                self._last_action_poll = now
                self.actions.refresh()
                self._sync_action_goal()
        finally:
            self._refresh_lock.release()

    def _sync_action_goal(self):
        frame = self.workspace.snapshot().get('goal')
        if not frame or frame.get('kind') != 'reminder':
            return
        action = self.actions.status()
        state = action.get('status', 'unknown')
        mapped = {'draft': 'awaiting_confirmation', 'proposed': 'awaiting_confirmation',
                  'idle': 'cancelled', 'expired': 'failed', 'running': 'scheduled'}.get(state, state)
        if mapped not in {'proposed', 'awaiting_confirmation', 'submitting', 'scheduled',
                          'cancelled', 'unknown', 'failed', 'completed'}:
            mapped = 'unknown'
        try:
            self.workspace.set_goal_state(mapped, 'task_receipt' if action.get('task') else 'confirmation',
                                          goal_id=frame['id'], revision=frame['revision'])
        except ValueError:
            # A concurrent new goal or expired episode wins over an old receipt.
            pass

    def attention(self, ref, revision):
        # Refresh before resolving a browser reference, then validate its exact
        # revision. A client cannot inject a label, person or camera observation.
        self._read_context()
        with self._context_lock:
            self.workspace.select_attention(ref, revision)
            self.cancel()
            self.actions.discard_draft()
            self.situation.clear()
            selected = self.workspace.snapshot()['attention']
            self.workspace.set_goal('inspect_object', '一起了解：' + selected['label'], 'touch')
        return self.status()

    def confirm_action(self, token):
        if type(token) is not str or not 0 < len(token) <= 128:
            raise ValueError('invalid confirmation')
        with self._action_control_lock:
            self.cancel()
            result = self.actions.confirm(token)
        self._sync_action_goal()
        if result.get('accepted') is not True:
            raise RuntimeError('这次确认未获有效回执；请核对当前提议和任务状态，不会重复创建')
        return self.status()

    def cancel_action(self):
        self.cancel()
        self.actions.cancel()
        self._sync_action_goal()
        return self.status()

    def reset(self):
        with self._action_control_lock, self._context_lock:
            self.actions.discard_draft(require_resettable=True)
            self.cancel()
            self.situation.clear()
            self.workspace.clear()
        return self.status()

    def output_receipt(self, job_id, state, *, terminal=False):
        if state not in {'queued', 'synthesizing', 'playing', 'completed', 'failed',
                         'cancelled', 'interrupted', 'expired', 'unknown'}:
            return
        with self._context_lock:
            if job_id != self.delivery.get('job_id'):
                return
            if state in {'completed', 'failed', 'cancelled', 'interrupted', 'expired'} and not terminal:
                return
            self.delivery = {'job_id': job_id, 'state': state, 'confirmed_terminal': bool(terminal)}

    def register_output(self, job_id, turn_token):
        with self._context_lock:
            if turn_token is not None and turn_token == self._delivery_turn:
                self.delivery = {'job_id': job_id, 'state': 'queued', 'confirmed_terminal': False}

    def cancel(self):
        # Revoke semantic authority before waiting for native cleanup.
        with self._context_lock:
            self._bump_control()
            self.situation.invalidate()
        result = self.kiosk.cancel_current_turn()
        return {**self.status(), **result}

    def select(self, task_id):
        if task_id is not None and task_id not in {row['id'] for row in TASKS}:
            raise ValueError('unknown task')
        with self._context_lock:
            self.cancel()
            self.actions.discard_draft()
            self.workspace.clear_attention()
            self.situation.clear() if task_id is None else self.situation.select(task_id)
            if task_id is not None:
                row = next(row for row in TASKS if row['id'] == task_id)
                self.workspace.set_goal('conversation', row['title'], 'touch')
            else:
                self.workspace.set_goal('conversation', '继续当前对话', 'touch')
        return self.status()

    def offers(self, enabled):
        if type(enabled) is not bool:
            raise ValueError('enabled must be boolean')
        with self._context_lock:
            self.cancel()
            self.situation.set_offers(enabled)
        return self.status()

    def decline(self):
        with self._context_lock:
            self.cancel()
            self.situation.decline()
        return self.status()

    def _inspect_existing_visual(self, generation, current):
        """Use only an already running camera; never call preview.start()."""
        kiosk = self.kiosk
        if not kiosk.scene_lock.acquire(blocking=False):
            raise RuntimeError('视觉模型正在处理，请稍后再问')
        try:
            with kiosk.vision_lock:
                if not kiosk.vision_enabled or kiosk.vision_generation != generation or not current():
                    raise RuntimeError('视觉权限或当前任务已经变化')
            result = kiosk._detect_latest()
            counts = Counter(row['label'] for row in result.get('detections', []))
            with kiosk.vision_lock:
                if not kiosk.vision_enabled or kiosk.vision_generation != generation or not current():
                    raise RuntimeError('视觉权限或当前任务已经变化')
                kiosk._submit_object_context(result, dict(counts))
        finally:
            kiosk.scene_lock.release()

    @staticmethod
    def _observation_rows(packet):
        observations = packet.get('observations', {})
        rows = {}
        for kind in ('people', 'objects', 'posture', 'affect'):
            value = observations.get(kind)
            values = value if isinstance(value, list) else [value] if isinstance(value, dict) else []
            if values:
                rows[kind] = values
        return rows

    @staticmethod
    def _observation_signature(rows):
        # Refresh timestamps/confidence do not change a still-fresh semantic
        # fact. Source, actor track and asserted content do; new people/objects
        # in an already used modality also invalidate an old count/scene claim.
        fields = ('source', 'track_id', 'label', 'count', 'state', 'top_label', 'provider', 'inference')
        values = []
        for row in rows:
            value = tuple((key, row.get(key)) for key in fields)
            value += (('reasons', tuple(sorted(row.get('reasons', [])))),)
            values.append(value)
        return sorted(values, key=repr)

    @staticmethod
    def _language_only_turn(message, history):
        """Text-first dialogue; explicit scene/deictic input keeps grounding.

        This chooses context only, never answers or executes a user request.
        """
        value = message.lower().strip()
        if perception_query(value) == 'identity':
            # Select a text-only model input, NOT an identity answer. The model
            # still decides whether to request the host-owned local face read.
            return True
        if perception_query(value) != 'chat' or re.search(
                r'这个|那个|这些|那些|它|他|她|这里|那里|这边|那边|眼前|现场|'
                r'看起来|看着|桌上|手里|身边|周围|还在|在不在|是否在|'
                r'什么情况|发生什么|怎么了|杯子|水杯|物体|人脸|脸红|手.{0,5}抖|衣服.{0,5}颜色|'
                r'\b(?:this|that|these|those|it|they|he|she|here|there|see|look|visible)\b', value):
            return False
        number = r'(?:[-+]?\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万点负]+)'
        operator = r'(?:加上?|减去?|乘以?|除以?|[+＋×÷*/−-]|plus|minus|times|divided\s+by)'
        expression = number + r'\s*(?:' + operator + r'\s*' + number + r'\s*)+'
        prefix = r'(?:(?:现在)?(?:请问|请|帮我|请帮我)?\s*(?:算(?:一下|算)?|计算(?:一下)?)?\s*|what\s+is\s+)'
        ending = r'(?:(?:等于|结果是|是)\s*(?:几|多少|什么)?|=\s*\??)?\s*(?:呢|吗|啊|呀)?[？?。.!！]*'

        def arithmetic(text):
            return re.fullmatch(prefix + expression + ending, text.lower().strip()) is not None

        if arithmetic(value):
            return True
        followup = (r'(?:那|那么)?\s*(?:再|然后|接着)\s*' + operator + r'\s*' + number
                    + r'\s*(?:(?:等于|结果是|是)\s*(?:几|多少|什么)?)?\s*(?:呢|吗|啊|呀)?[？?。.!！]*')
        if re.fullmatch(followup, value):
            # Follow-up inherits a bounded contiguous arithmetic topic, not
            # the most recent number mentioned in a scene or reminder.
            for row in reversed(history):
                if row.get('role') != 'user':
                    continue
                previous = row.get('content', '')
                if arithmetic(previous):
                    return True
                if re.fullmatch(followup, previous.lower().strip()) is None:
                    return False
            return False
        if re.fullmatch(r'(?:你好|您好|嗨|谢谢(?:你)?|早上好|晚上好|hello|hi|thank you)[，,。.!！?？\s]*', value):
            return True
        # Clearly authored knowledge/explanation requests need no live scene.
        # Broad/deictic forms ("解释这个") were excluded above; unknown stays
        # grounded rather than making the model self-declare dependencies.
        # Ordinary dialogue is text-first. Do not make every novel question
        # depend on a person moving or a camera TTL merely because it was not
        # on a phrase allowlist. Explicit scene/deictic topics above retain
        # the existing grounded path; no extra detector/model is run here.
        return True

    @staticmethod
    def _language_only_projection(reasoning, workspace):
        """Remove unused volatile facts BEFORE inference; leave host RAM intact."""
        reasoning, workspace = deepcopy(reasoning), deepcopy(workspace)
        reasoning['observations'] = {'people': [], 'objects': [], 'posture': None, 'affect': None}
        reasoning['recent_semantic_changes'] = []
        for key in ('identity', 'people', 'objects', 'posture', 'affect', 'fusion'):
            reasoning['source_states'][key] = {'state': 'unknown'}
        reasoning['omitted'].append('perception:not_required_for_text_topic')
        workspace['actors']['visible_people'] = []
        workspace['actors']['presence_state'] = 'unknown'
        workspace['attention'] = None
        if workspace.get('goal') and workspace['goal']['kind'] != 'conversation':
            workspace['goal'] = None
        workspace['recent_events'] = [row for row in workspace['recent_events']
                                      if row['kind'] in {'input_received', 'reply_recorded'}]
        return reasoning, workspace

    def _revalidate_perception(self, request, original, vision_basis, current):
        """Re-read only anonymous metadata; never renew evidence or capture.

        All perceptual facts supplied to the model are conservatively treated as
        dependencies. Robot/output/resource snapshots are deliberately excluded:
        their short housekeeping TTL must not invalidate ordinary task help.
        """
        rows = self._observation_rows(original)
        if not rows:
            return True
        if not current():
            return False
        now = time.time()
        original_fresh = True
        for values in rows.values():
            for row in values:
                observed, expires = row.get('observed_at'), row.get('expires_at')
                if (type(observed) not in (int, float) or type(expires) not in (int, float)
                        or not observed <= now < expires
                        or not math.isfinite(observed) or not math.isfinite(expires)):
                    original_fresh = False

        def visual_current():
            if vision_basis is None:
                return True
            with self.kiosk.vision_lock:
                return (vision_basis['enabled'] is True and self.kiosk.vision_enabled
                        and self.kiosk.vision_generation == vision_basis['generation'])

        # This read runs even if an original observation expired so diagnostics
        # describe the last recheck snapshot, not a falsely live old model input.
        try:
            fresh = assemble_turn_context(request, self.kiosk._perception_context(), None, [])
            latest = build_reasoning_packet(request, fresh)
        except Exception:
            self.source_states = {kind: 'unavailable' for kind in rows}
            self.context_observed_at = None
            self.workspace.ingest({})
            return False
        self.source_states = {key: value['state'] for key, value in latest['source_states'].items()}
        self.context_observed_at = latest['basis'].get('captured_at')
        self.workspace.ingest(latest)
        if not original_fresh or not current() or not visual_current():
            return False
        latest_rows = self._observation_rows(latest)
        return all(self._observation_signature(values)
                   == self._observation_signature(latest_rows.get(kind, []))
                   for kind, values in rows.items())

    def voice(self, *, control=None):
        if control is not None:
            # A cancel may overtake a delayed HTTP voice request. Reject its
            # stale generation before begin() can invalidate a newer ticket.
            # Kiosk repeats this check at actual microphone admission.
            with self.kiosk.turn_manager.lock:
                current = self.kiosk.turn_manager.snapshot()
                if (control.get('control_epoch') != current['control_epoch']
                        or control.get('control_revision') != current['control_revision']):
                    return {'accepted': False, 'reason': 'stale_control', 'local_only': True}
                ticket = self.situation.begin('voice')
        else:
            ticket = self.situation.begin('voice')
        try:
            options = {'agent_control': control} if control is not None else {}
            value = self.kiosk.voice_turn(agent_ticket=ticket, **options)
            # Transcription is single-use input, not a public diagnostic.
            value.pop('transcript', None)
            return value
        finally:
            if self.situation.current(ticket):
                self.situation.invalidate()

    @staticmethod
    def _answer_value(message):
        match = re.fullmatch(r'(?:我选|我觉得是|答案是|答案为|是)?\s*([0-9]{1,3}|[一二三四五六七八九十])\s*[。.!！]?', message)
        if not match:
            return None
        value = match.group(1)
        return {'一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
                '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'}.get(value, value)

    @staticmethod
    def _resource_busy(packet):
        # Existing LocalLLM/Runtime admission remains authoritative. This is an
        # additional cheap refusal when a fresh shared budget is exhausted.
        return any(row.get('kind') == 'resources' and row.get('state') == 'present'
                   and type(row.get('data', {}).get('cpu_available')) in (int, float)
                   and row['data']['cpu_available'] <= 0
                   for row in packet.get('robot_state', []))

    @staticmethod
    def _unconfirmed_object_presence(message, reasoning):
        # A narrow availability boundary, not an object detector or a general
        # intent model. Only literal current-presence questions about objects
        # use it; general uses, plans, people, quotes and composite requests do
        # not. Never turn unavailable evidence into a presence/absence claim.
        state = reasoning.get('source_states', {}).get('objects', {})
        source_state = state.get('state')
        if (source_state not in {'stale', 'unknown', 'disabled', 'unavailable'}
                or reasoning.get('observations', {}).get('objects')):
            return None
        query = (r'(?:请问)?(?:(?:刚才|之前)(?:的|那个|那只|那本)?|那个|那只|那本)?'
                 r'(?:杯子|水杯|书本|书|笔|玩具|物体|东西)[，,\s]*'
                 r'(?:现在|目前|此刻)?(?:还在(?:那里|这里|那儿|这儿)?|在不在|是否还在|'
                 r'还能看见|还能看到|还看得到)(?:吗|么|呢)?[？?。]?')
        if not re.fullmatch(query, message):
            return None
        basis = {'stale': '我这次拿到的物体线索已过期',
                 'unknown': '我这次没有可靠的当前物体线索',
                 'disabled': '这次的物体感知来源未开启',
                 'unavailable': '这次的物体感知来源不可用'}[source_state]
        return {'status': 'ok', 'intent': 'clarify',
                'response': basis + '，不能据此确认它现在是否还在。',
                'backend': 'object-evidence-boundary', 'model_calls': 0}

    def _explicit_action(self, message, input_type):
        draft = self.actions.propose(message, input_type=input_type)
        if draft:
            self.workspace.set_goal('reminder', draft['summary'], input_type,
                                    slots={'minutes': draft['minutes'], 'topic': draft['topic']})
            self._sync_action_goal()
            return {'intent': 'chat', 'response': '我准备' + draft['summary'] + '。请在屏幕确认，现在还没有创建。',
                    'backend': 'typed-reminder-draft', 'model_calls': 0}
        if re.fullmatch(r'(?:请)?取消(?:这个|刚才的)?提醒[。！!]?', message):
            self.actions.cancel()
            self._sync_action_goal()
            state = self.actions.status().get('status')
            text = ('这个提醒已取消。' if state == 'cancelled' else
                    '目前没有待确认的提醒。' if state == 'idle' else
                    '取消结果还未确认，请看实际任务状态；我不会假定已经取消。')
            return {'intent': 'chat', 'response': text, 'backend': 'explicit-action-control', 'model_calls': 0}
        if re.fullmatch(r'(?:这个|刚才的)?(?:提醒|任务)?(?:已经)?(?:好了没|做好了吗|安排好了吗|完成了吗|取消了吗|预约了吗)[。？?]?', message):
            state = self.actions.status().get('status')
            quiz = self.situation.snapshot()
            text = {'scheduled': '任务服务已确认预约，尚不能当作已经执行。',
                    'running': '任务服务报告正在执行，还没有完成回执。',
                    'completed': '任务服务报告已完成；我没有核实你是否实际听到声音。',
                    'cancelled': '取消已获确认。', 'failed': '任务没有完成。',
                    'proposed': '目前只是提议，尚未确认，也没有创建任务。',
                    'idle': ('这道题的答案已核对通过。' if quiz.get('task') and quiz.get('state') == 'completed' else
                             '你想确认的是哪件事？当前没有已提交提醒的执行回执。')}.get(state, '当前结果尚未确认，我不能声称已经完成或取消。')
            return {'intent': 'chat', 'response': text, 'backend': 'owned-action-receipt', 'model_calls': 0}
        if re.search(r'提醒|闹钟|倒计时|定时|(?:改成|改为).{0,10}分钟', message):
            return {'intent': 'clarify', 'response': '目前可以确认休息、喝水或活动提醒。请说“十分钟后提醒我休息”，再在屏幕确认；我还没有新建任务。',
                    'backend': 'unsupported-action-boundary', 'model_calls': 0}
        if re.match(r'^(?:请|帮我|给我|你能|可以|能不能|麻烦你)?(?:打开|开启|关闭|播放|录音|删除|保存|发送|启动|停止|重启|部署|记住)', message):
            return {'intent': 'clarify', 'response': '这次没有执行设备或资料操作。请使用对应的明确控制入口；我不能把聊天回复当作执行授权。',
                    'backend': 'unsupported-action-boundary', 'model_calls': 0}
        return None

    @staticmethod
    def _unverified_execution_claim(text):
        # Free-form model prose is never an execution receipt, even when its
        # JSON schema is valid. Conservative refusal beats fabricated success.
        verbs = r'创建|设置|安排|预约|取消|开启|打开|关闭|录音|保存|删除|发送|播放|调用|完成|更新|启动|停止|部署|读取|记住'
        return bool(re.search(r'(?:已经|成功|正在|已为你|已帮你|已|现在就).{0,18}(?:' + verbs + r')', text)
                    or re.search(r'(?:' + verbs + r').{0,12}(?:成功|完毕|完成|好了|好啦|好啦|了)', text)
                    or re.search(r'(?:我会|我将|到时候|稍后|一会儿|届时|分钟后).{0,18}(?:提醒你|叫你|通知你|播放|保存)', text)
                    or re.search(r'(?:提醒|闹钟|任务|预约|相机|摄像头|录音|播放|资料).{0,12}(?:搞定|妥当|就绪|办好)', text)
                    or re.search(r'\b(?:done|scheduled|cancelled|canceled|created|saved|deleted|opened|closed|started|stopped|sent|completed)\b', text, re.I)
                    or re.search(r'\b(?:alarm|reminder|task)\b.{0,30}\bset\b', text, re.I))

    @staticmethod
    def _arithmetic_reply_consistent(proposal, text):
        """Reject an explicit wrong numeric conclusion; never rewrite prose."""
        numeric=r'[-+]?\d+(?:\.\d+)?'
        conclusions=re.findall(r'(?:等于|结果(?:是|为)|答案(?:是|为)|算成)\s*('+numeric+r')',text)
        expected=proposal.get('result')
        if expected is None:
            return not conclusions
        numbers=re.findall(numeric,text)
        conclusion=conclusions[-1] if conclusions else numbers[0] if len(numbers)==1 else None
        return conclusion is None or Decimal(conclusion)==Decimal(expected)

    def chat(self, message, *, input_type='text', turn_token=None, ticket=None):
        if (not isinstance(message, str) or not 0 < len(message.strip()) <= 1000
                or input_type not in ('text', 'voice')):
            raise ValueError('invalid situated input')
        message = message.strip()
        kiosk = self.kiosk
        if not kiosk.lock.acquire(blocking=False):
            raise RuntimeError('芽芽正在处理上一件事，请先等一下或点停止')
        started = time.monotonic()
        prepared_media_id = None
        budget_ticket = None
        goal_basis = None
        used_model = False
        arithmetic_proposal = None
        try:
            self.last_timing = {}
            self.last_face_context = None
            if ticket is not None and not self.situation.current(ticket):
                return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
            kiosk.quiet.interrupt()
            turn_token = turn_token or kiosk.turn_manager.begin('thinking')
            turn_control = kiosk.turn_manager.snapshot()
            kiosk._pending_decision = None
            # A voice ticket is bound before capture; text binds to the current
            # selected task before any potentially blocking source/model work.
            ticket = ticket or self.situation.begin(input_type)
            with self._context_lock:
                control_revision = self._control_revision
            def authority_current(*, finished=False):
                with self._context_lock:
                    # finish() deliberately consumes the pending ticket but
                    # leaves its revision unchanged. Delivery acceptance must
                    # fence that revision, not require an already-used ticket.
                    situation_current = (self.situation.snapshot()['revision'] == ticket['revision']
                                         if finished else self.situation.current(ticket))
                    turn_now = kiosk.turn_manager.snapshot()
                    turn_current = ((turn_now['control_epoch'], turn_now['control_revision'])
                                    == (turn_control['control_epoch'], turn_control['control_revision'])
                                    if finished else kiosk.turn_manager.current(turn_token))
                    if not (self._control_revision == control_revision
                            and situation_current and turn_current):
                        return False
                    goal = self.workspace.snapshot().get('goal')
                    return goal_basis is None or bool(goal and (goal['id'], goal['revision']) == goal_basis)
            def current():
                return authority_current() and (budget_ticket is None or self.budget.current(budget_ticket))
            if not current():
                return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
            budget_ticket = self.budget.begin(current=authority_current)
            if budget_ticket is None:
                raise RuntimeError('本地模型还在结束上一轮，请稍后重试')
            with self._action_control_lock, self._context_lock:
                if not current():
                    return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
                history = self.workspace.dialogue()
                self.workspace.input(message, input_type)
            kiosk.state.update({'phase': 'thinking', 'visual_style': 'thinking',
                                'primary_text': '我在想这件事', 'secondary_text': '',
                                'producer': 'conversation'})
            decision = None
            safety = self.safety.evaluate(message) if self.safety else None
            if safety is not None and not safety.allow_model:
                decision = {'intent': 'clarify', 'response': safety.response,
                            'backend': 'safety-policy', 'model_calls': 0}
            # Explicit refusal is an authoritative control, never an inferred
            # emotional preference and never a durable personal-profile write.
            if decision is None and re.fullmatch(r'(?:我)?(?:现在|这次|先)?(?:不需要|不用|不要)(?:你的|你|再)?(?:了|帮助|帮忙|提示|帮我|再提示)?(?:[，,](?:让我)?自己(?:试试|做|来))?[。！!，,]?', message):
                with self._context_lock:
                    if not current() or self.situation.decline(ticket=ticket) is None:
                        return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
                    ticket = self.situation.begin(input_type)
                decision = {'intent': 'decline_help', 'response': '好，我先不提示。需要时你再叫我。',
                            'backend': 'explicit-control', 'model_calls': 0}
            situation = self.situation.packet(ticket)
            if situation is None:
                return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
            task = situation.get('task') or {}
            if decision is None:
                with self._action_control_lock, self._context_lock:
                    if not current():
                        return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
                    decision = self._explicit_action(message, input_type)
                    if decision and decision.get('backend') == 'typed-reminder-draft' and task:
                        self.situation.clear()
                        ticket = self.situation.begin(input_type)
                        situation = self.situation.packet(ticket)
                        task = {}
            with self._context_lock:
                if not current():
                    return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
                if decision is None and not self.workspace.snapshot().get('goal'):
                    # Raw input stays bounded RAM data, not public goal text.
                    self.workspace.set_goal('conversation', '继续当前对话', input_type)
            if decision is None and re.search(r'谁.{0,8}说话|说话.{0,8}谁', message):
                # No speaker-binding provider exists on this surface. A text
                # model must not turn missing evidence into an absence claim.
                decision = {'intent': 'clarify', 'response': '当前还不能可靠确认说话者是谁。镜头里出现的人，也不能直接当作说话的人。',
                            'backend': 'speaker-binding-boundary', 'model_calls': 0}
            # Identity questions are understood by the same dialogue model as
            # other natural input. A literal-question shortcut must not bypass
            # its contextual choice of whether this turn needs a local read.
            if (decision is None and not task and re.fullmatch(
                    r'(?:这个|这道题|这一题|这一步)(?:我)?(?:不会|怎么做|怎么想|怎么理解)[。？?]?', message)):
                decision = {'intent': 'clarify', 'response': '你想一起看哪一道题？先点选题目，或者把题目内容告诉我。',
                            'backend': 'missing-task-boundary', 'model_calls': 0}
            # Only an explicit, single-scope hint request can take this fast
            # path. "不会/为什么" alone, negated hint requests and composite
            # instructions still need the bounded model's intent judgement.
            explicit_hint = bool(re.fullmatch(
                r'(?:这个我不会[，,]\s*)?(?:(?:能|可以|能不能|可不可以)\s*)?'
                r'(?:先\s*)?(?:给(?:我)?(?:一点|一个|点|个)?(?:小)?提示|提示一下)'
                r'(?:吗|么|吧)?[。？?！!]?', message))
            if decision is None and explicit_hint:
                if task:
                    decision = {'intent': 'help_task', 'response': task['hint'],
                                'backend': 'explicit-task-hint', 'model_calls': 0}
                else:
                    decision = {'intent': 'clarify', 'response': '你想一起看哪一道题？先点选题目，或者把题目内容告诉我。',
                                'backend': 'missing-task-boundary', 'model_calls': 0}
            candidate = self._answer_value(message)
            if task.get('id') == 'reading-rain-1':
                clean = re.sub(r'[，。！？!?\s]', '', message)
                if clean in {'挡雨', '因为下雨了', '下雨了', '为了挡雨', '怕淋雨', '不被淋湿', '因为下雨了要用伞挡雨'}:
                    candidate = str(task['answer'])
            if decision is None and candidate is not None and task:
                correct = candidate == str(task.get('answer'))
                with self._context_lock:
                    if not current() or self.situation.feedback(correct, ticket=ticket) is None:
                        return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
                    goal = self.workspace.snapshot().get('goal')
                    if correct and goal and goal.get('kind') == 'conversation' and goal.get('title') == task['title']:
                        self.workspace.set_goal_state('completed', 'answer_checked')
                    ticket = self.situation.begin(input_type)
                    situation = self.situation.packet(ticket)
                answer = ('答对啦！要换一题，还是说说你是怎么想到的？' if correct else
                          '这次还不对。你可以再试一下，也可以问我怎么想。')
                offers = (situation or {}).get('offers', {})
                if not correct and offers.get('eligible'):
                    answer = '我可以给你一个小提示，要听听吗？'
                decision = {'intent': 'answer_task', 'response': answer,
                            'backend': 'verified-task-answer', 'model_calls': 0,
                            'offer_help': not correct and bool(offers.get('eligible')),
                            'correct': correct}
            with self._context_lock:
                if not current():
                    return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
                if decision is None and not task:
                    arithmetic_proposal = self.arithmetic.prepare(message)
                if arithmetic_proposal is None:
                    # Topic changes, selected tasks and explicit controls never
                    # inherit an unrelated calculation as an implicit subject.
                    self.arithmetic.invalidate()
            if decision is None:
                if not current():
                    return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
                # Use the existing, freshness-filtered semantic projection. A
                # visible face is deliberately not a microphone/operator ID.
                acquired = time.monotonic()
                request = {'user_text': message, 'input_type': input_type,
                           'session_id': 'starbuddy.situated',
                           'messages': [{'role': row['role'], 'text': row['content']} for row in history],
                           'memory': {}, 'dialogue_window': 6}
                reasoning = self._read_context(request)
                workspace = self.workspace.packet()
                if arithmetic_proposal is not None or (
                        self._language_only_turn(message, history)):
                    # An earlier selected object is not the subject of every
                    # later sentence. Explicit/deictic scene requests remain
                    # grounded; a new clear text topic keeps its fast route.
                    reasoning, workspace = self._language_only_projection(reasoning, workspace)
                context_ms = round((time.monotonic() - acquired) * 1000, 2)
                with kiosk.vision_lock:
                    visual_enabled = kiosk.vision_enabled
                    vision_generation = getattr(kiosk, 'vision_generation', None)
                # This is a read of an already-enabled service's metadata,
                # never permission to start a camera, enroll or read profiles.
                allowed = ['perception.visible_identity']
                if perception_query(message) == 'scene' and visual_enabled:
                    allowed.append('perception.scene')
                used_environment = any(row.get('source') == 'camera.environment'
                    for rows in self._observation_rows(reasoning).values() for row in rows)
                vision_basis = {'enabled': visual_enabled, 'generation': vision_generation} if used_environment else None
                packet = {'situation': situation, 'reasoning': reasoning,
                          'workspace': workspace,
                          'user_input': message, 'allowed_capabilities': allowed}
                if arithmetic_proposal is not None:
                    # A verified value is model grounding, NOT the spoken
                    # reply. The model still understands/composes this turn.
                    packet['verified_calculation'] = {
                        'result': arithmetic_proposal['result'], 'error': arithmetic_proposal['error'],
                        'source': 'host_exact_arithmetic', 'execution_authority': False}
                goal = packet['workspace'].get('goal')
                goal_basis = (goal['id'], goal['revision']) if goal else None
                selected_attention = packet['workspace'].get('attention')
                decision = self._unconfirmed_object_presence(message, reasoning) if not task else None
                if decision is not None:
                    pass
                elif re.fullmatch(r'(?:请问)?(?:这个|它)(?:是|叫)?什么(?:东西)?[。？?]?', message):
                    label = (selected_attention or {}).get('label')
                    response = ('现在选中的是“' + task['title'] + '”这道题。你想先听题意，还是一起想第一步？' if task else
                                '你选中的是“' + label + '”这一类物体；这是识别线索，不是具体个体的身份。' if label else
                                '请先在“当前线索”里点选物体类别，我还不能确定你说的“这个”是哪一个。')
                    decision = {'status': 'ok', 'intent': 'help_task' if task else 'chat' if label else 'clarify',
                                'response': response, 'backend': 'shared-attention', 'model_calls': 0}
                else:
                    cloud_ready = getattr(self.reasoner.runtime, '_cloud_ready', lambda: False)() if hasattr(self.reasoner, 'runtime') else False
                    if not self.budget.claim(budget_ticket, 'model', resource_busy=self._resource_busy(reasoning) and not cloud_ready):
                        raise RuntimeError('本地算力正忙，本轮没有调用理解模型；可以稍后再问')
                    used_model = True
                    decision = self.reasoner.analyze(packet, current=current)
                if decision.get('status', 'ok') != 'ok':
                    if not current():
                        return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
                    raise RuntimeError('这次理解没有完成，请稍后重试；没有执行任何新动作')
                if decision.get('requested_capability') == 'perception.visible_identity':
                    if 'perception.visible_identity' not in allowed or not current():
                        return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
                    grounded = self._face_context_reply(message, requested=True)
                    # Keep inference accounting/route. The model proposed only
                    # this read; the host exclusively owns the private answer.
                    decision.update(response=grounded['response'], intent='chat',
                                    private_context=True, identity_evidence=grounded['identity_evidence'])
                    decision['backend'] = decision.get('backend', 'local-bounded') + '+local-face-context'
                if arithmetic_proposal is not None:
                    decision['backend'] = decision.get('backend', 'local-bounded') + '+verified-arithmetic'
                    if arithmetic_proposal['result'] is None:
                        decision['intent'] = 'clarify'
                    if not self._arithmetic_reply_consistent(arithmetic_proposal, decision.get('response','')):
                        raise RuntimeError('这次计算说明没有与已核验结果一致，未朗读错误结论；可以重新问完整算式')
                if decision.get('requested_capability') == 'perception.scene':
                    if 'perception.scene' not in allowed or not current():
                        raise RuntimeError('本轮没有获准开启新的视觉采集')
                    if not self.budget.claim(budget_ticket, 'inspection', resource_busy=self._resource_busy(reasoning)):
                        raise RuntimeError('本轮精查预算不可用，没有再次调用视觉模型')
                    self._inspect_existing_visual(vision_generation, current)
                    if not current():
                        return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
                    # One bounded inspection, then only the existing grounded
                    # semantic summary; do not enter another reasoning loop.
                    context = assemble_turn_context(request, kiosk._perception_context(), None, [])
                    latest = build_reasoning_packet(request, context)
                    objects = latest['observations']['objects']
                    counts = {row['label']: row['count'] for row in objects}
                    decision['response'] = (kiosk._compact_counts(counts) + '。' if counts else
                                            '这次还没有得到可靠的物体结果，我先不猜。')
                    decision['backend'] += '+bounded-vision'
                    # The deterministic inspection summary now depends on this
                    # new admitted snapshot, not the older model-input facts.
                    reasoning = latest
                    if any(row.get('source') == 'camera.environment'
                           for rows in self._observation_rows(reasoning).values() for row in rows):
                        vision_basis = {'enabled': visual_enabled, 'generation': vision_generation}
                recheck_started = time.monotonic()
                # A local identity answer has its own stricter fresh/unique
                # evidence check, repeated immediately before Output below.
                perception_current = (True if decision.get('private_context') else
                    self._revalidate_perception(request, reasoning, vision_basis, current))
                if not current():
                    return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
                if not perception_current:
                    # Do not replay an old scene assertion or run a second
                    # inference. An explicit selected task remains trustworthy.
                    if task and decision.get('intent') == 'help_task':
                        decision['response'] = task['hint']
                    else:
                        decision['intent'] = 'clarify'
                        decision['response'] = '当前现场信息还不能确认，我先不沿用刚才的判断。你想让我确认哪一部分？'
                    decision['context_changed'] = True
                if selected_attention is not None and not decision.get('private_context'):
                    attention_now = self.workspace.snapshot().get('attention')
                    if (attention_now is None or attention_now['ref'] != selected_attention['ref']
                            or time.time() >= selected_attention['expires_at']):
                        decision.update(intent='clarify', context_changed=True,
                                        response='刚才选中的物体线索已失效，请重新点选，我先不沿用旧对象。')
                self.last_timing = {'context': context_ms, 'reasoning': decision.get('elapsed_ms')}
                self.last_timing['context_recheck'] = round((time.monotonic() - recheck_started) * 1000, 2)
            if not current():
                return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
            text = decision.get('response')
            if not isinstance(text, str) or not 0 < len(text.strip()) <= 600:
                raise RuntimeError('理解结果未通过校验，请换个说法再试')
            intent = decision.get('intent', 'clarify')
            operational_claim = re.search(
                r'提醒|闹钟|预约|摄像头|相机|录音|资料|文件|网络|Wi-Fi|麦克风|音量|服务|部署|保存|删除|发送', text)
            if (used_model and self._unverified_execution_claim(text)
                    and (decision.get('response_mode') != 'natural_dialogue' or operational_claim)):
                text = '我还没有执行新动作，也不能从聊天结果确认完成。请查看提议和实际任务回执。'
                intent = 'clarify'
            if ((intent == 'answer_task' or (intent == 'help_task' and not self.workspace.snapshot().get('goal')))
                    and not task):
                text = '你想一起看哪一道题？先点选题目，或者把题目内容告诉我。'
                intent = 'clarify'
            if intent == 'decline_help' and decision.get('backend') != 'explicit-control':
                # A proposal is not an explicit preference mutation. The user
                # can confirm through the dedicated decline control.
                text = '你想先自己试试、暂时不需要提示，对吗？可以点“先不用提示”。'
                intent = 'clarify'
            if (task and intent in {'help_task', 'answer_task'}
                    and task.get('id') in {'math-apples-1', 'pattern-even-1'}):
                known_numbers = set(re.findall(r'\d+', task['prompt'] + ' '.join(task['steps'])))
                if set(re.findall(r'\d+', text)) - known_numbers:
                    text = task['hint']
            if task and intent == 'help_task' and re.search(r'提示|线索|先别告诉我答案|不要直接给答案', message):
                # Respect the requested pedagogical scope even if a language
                # model tries to reveal the answer. This does not rewrite its
                # intent judgement or claim general semantic verification.
                text = task['hint']
            action = 'offer_help' if decision.get('offer_help') else (
                'clarify' if intent == 'clarify' else 'respond')
            state_decision = {'action': action, 'intent': intent,
                              'target': task.get('id') or 'current_turn',
                              'reason': 'uncertain' if decision.get('context_changed') else
                                        'user_declined' if intent == 'decline_help' else
                                        'explicit_request' if decision.get('backend') == 'explicit-task-hint' else
                                        'answer_checked' if (candidate is not None and task
                                            or arithmetic_proposal is not None) else 'model_decision'}
            if decision.get('private_context'):
                # A metadata read may have been delayed; re-read before Output.
                # Identity remains a local reply, not an input to either LLM.
                latest_face = self._face_context_reply(message, requested=True, recheck=True)
                text = (latest_face or {}).get('response') or '这次还没认出镜头里的人。我们可以先聊天。'
            audio_started = time.monotonic()
            preferences = getattr(kiosk, 'preferences', None)
            speed = preferences.snapshot()['values']['speed'] if preferences is not None else 1.0
            prepared_media_id = self.fixed_audio.prepare(text.strip(), speed=speed)
            self.last_timing['fixed_audio_import'] = round((time.monotonic() - audio_started) * 1000, 2)
            if not current():
                return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
            if not self.situation.finish(ticket, state_decision):
                return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
            if not kiosk.turn_manager.current(turn_token):
                return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
            prepared_options = {'prepared_media_id': prepared_media_id} if prepared_media_id is not None else {}
            with self._context_lock:
                self._delivery_turn = turn_token
            cloud_used = str(decision.get('backend', '')).startswith('cloud-text-api')
            spoken = kiosk._respond(text.strip(), primary=text.strip(), secondary=(
                                    '文字 API 回应，不发送摄像头或个人档案' if cloud_used else '本轮本地处理，不写入长期记忆'),
                                    visual_style='gentle', turn_token=turn_token, **prepared_options)
            # A queue acknowledgement is not completion or evidence the user
            # heard anything. Only the existing owned Output watcher settles it.
            with self._context_lock:
                job_id = spoken.get('playback_id')
                if self.delivery.get('job_id') != job_id or job_id is None:
                    self.delivery = {'job_id': job_id, 'state': 'queued' if job_id else 'unknown',
                                     'confirmed_terminal': False}
            with self._context_lock:
                # An Output call can race with Stop. Never turn its late
                # return into a new arithmetic base or fresh dialogue reply.
                if not authority_current(finished=True):
                    return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
                if arithmetic_proposal is not None and not self.arithmetic.commit(arithmetic_proposal):
                    return {'accepted': False, 'reason': 'cancelled', 'local_only': True}
                self.workspace.reply('我回应了本地人脸识别问题；姓名未发送给对话模型。'
                                     if decision.get('private_context') else text.strip())
            self.last_timing.update(total_to_output_queue=round((time.monotonic() - started) * 1000, 2),
                                    model_calls=decision.get('model_calls', 1) if used_model else 0)
            return {**spoken, 'accepted': True, 'intent': intent,
                    'local_only': not cloud_used,
                    'backend': decision.get('backend'), 'timing_ms': dict(self.last_timing),
                    'fixed_audio_used': prepared_media_id is not None,
                    'state': self.status()['state']}
        except Exception:
            with self._context_lock:
                self.arithmetic.invalidate()
            if self.situation.current(ticket):
                self.situation.invalidate()
            if turn_token and kiosk.turn_manager.current(turn_token):
                kiosk.cancel_current_turn()
            raise
        finally:
            if budget_ticket is not None:
                self.budget.finish(budget_ticket)
            if prepared_media_id is not None:
                self.fixed_audio.release(prepared_media_id)
            kiosk.lock.release()

    def _face_context_reply(self, message, *, requested=False, recheck=False):
        from gateway.display.face_dialogue import is_face_question, visual_identity_reply
        if requested is not True and not is_face_question(message):
            return None
        try:
            read_identity = getattr(self.kiosk, '_current_verified_identity', None)
            identity = read_identity() if callable(read_identity) else None
            snapshot = self.kiosk._perception_context()
            result = visual_identity_reply(message, snapshot, identity, requested=requested)
        except Exception:
            # Missing service/snapshot never resurrects a remembered identity.
            result = visual_identity_reply(message, None, None, requested=True)
        route = ((self.last_face_context or {}).get('route') if recheck else None)
        self.last_face_context = {**result['identity_evidence'],
                                  'route': route or ('llm_read_request' if requested else 'shortcut')}
        return result
