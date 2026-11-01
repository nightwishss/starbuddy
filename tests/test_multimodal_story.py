"""Full context/action stories with owned fakes, no models, sensors or playback."""
import json
import time
import unittest
from copy import deepcopy

from task_orchestrator.client import request_task_id
from test_situated_product import Kiosk


class TaskReceipts:
    def __init__(self):
        self.creates = []
        self.cancels = []
        self.task = None

    def create(self, utterance, **values):
        self.creates.append((utterance, deepcopy(values)))
        self.task = {'task_id': request_task_id(values['source_conversation_id'], values['request_id']),
                     'source_conversation_id': values['source_conversation_id'], 'state': 'scheduled'}
        return deepcopy(self.task)

    def get(self, task_id):
        assert task_id == self.task['task_id']
        return deepcopy(self.task)

    def cancel(self, task_id):
        self.cancels.append(task_id)
        assert task_id == self.task['task_id']
        self.task['state'] = 'cancelled'
        return deepcopy(self.task)


class MultimodalStories(unittest.TestCase):
    def setUp(self):
        from gateway.display.situated_actions import ReminderActionBroker
        self.kiosk = Kiosk()
        self.agent = self.kiosk.situated
        self.tasks = TaskReceipts()
        self.agent.actions = ReminderActionBroker(self.tasks)
        self.kiosk.reasoner.reply.update(intent='chat', response='我们接着刚才的事情想一想。')

    def scene(self):
        now = time.time()
        def row(**values):
            return {'source': 'camera.front', 'observed_at': now - .1, 'expires_at': now + 10,
                    'confidence': .95, 'provenance': {'provider': 'synthetic-story'}, **values}
        return {'people': [row(track_id='synthetic-person-a'), row(track_id='synthetic-person-b')],
                'objects': [row(label='cup', count=2), row(label='book', count=1)]}

    def install_scene(self):
        self.snapshot = self.scene()
        self.kiosk._perception_context = lambda: deepcopy(self.snapshot)
        self.agent.status(refresh=True)

    def draft(self, message='十分钟后提醒我休息'):
        answer = self.agent.chat(message)
        self.assertTrue(answer['accepted'])
        state = self.agent.status()['state']
        self.assertIn('确认', answer['answer'])
        self.assertEqual([], self.tasks.creates)
        self.assertEqual('reminder', state['workspace']['goal']['kind'])
        return state['action']['draft']['token']

    def test_passive_context_updates_never_run_a_model_or_capture(self):
        self.install_scene()
        state = self.agent.status()['state']
        self.assertEqual(2, len(state['workspace']['objects']))
        self.assertEqual(2, len(state['workspace']['actors']['visible_people']))
        self.assertEqual([], self.kiosk.reasoner.calls)
        self.assertEqual(0, self.kiosk.visual_calls)
        self.assertFalse(state['context_is_live'])

    def test_touch_object_then_voice_uses_shared_attention_not_face_identity(self):
        self.install_scene()
        state = self.agent.status()['state']['workspace']
        target = next(row for row in state['objects'] if row['label'] == 'book')
        self.agent.attention(target['ref'], state['revision'])
        answer = self.agent.chat('这个是什么？', input_type='voice')
        self.assertIn('book', answer['answer'])
        self.assertNotIn('cup', answer['answer'])
        self.assertEqual([], self.kiosk.reasoner.calls)
        actors = answer['state']['workspace']['actors']
        self.assertEqual('unknown', actors['microphone_speaker']['identity'])
        self.assertFalse(actors['microphone_speaker']['profile_authorized'])
        self.assertEqual('unproven', actors['cross_modal_identity_link'])

    def test_stale_or_injected_browser_reference_is_denied(self):
        self.install_scene()
        state = self.agent.status()['state']['workspace']
        with self.assertRaises(ValueError):
            self.agent.attention('profile-private-person', state['revision'])
        with self.assertRaises(ValueError):
            self.agent.attention(state['objects'][0]['ref'], state['revision'] - 1)
        self.assertEqual([], self.kiosk.reasoner.calls)

    def test_no_pointing_target_asks_instead_of_guessing(self):
        self.install_scene()
        result = self.agent.chat('这个是什么')
        self.assertEqual('clarify', result['intent'])
        self.assertIn('点选', result['answer'])
        self.assertEqual([], self.kiosk.reasoner.calls)

    def test_source_disabled_removes_old_shared_attention(self):
        self.install_scene()
        state = self.agent.status()['state']['workspace']
        self.agent.attention(state['objects'][0]['ref'], state['revision'])
        self.snapshot = None
        result = self.agent.chat('这个是什么')
        self.assertEqual('clarify', result['intent'])
        self.assertIsNone(result['state']['workspace']['attention'])
        self.assertEqual(0, self.kiosk.visual_calls)

    def test_followup_has_short_session_dialogue_and_explicit_goal(self):
        self.agent.chat('我想了解植物为什么需要阳光')
        self.agent.chat('那它在夜里怎么办？', input_type='voice')
        packet = self.kiosk.reasoner.calls[-1]
        history = packet['reasoning']['recent_dialogue']
        self.assertEqual('我想了解植物为什么需要阳光', history[0]['text'])
        self.assertEqual('assistant', history[1]['role'])
        self.assertEqual('conversation', packet['workspace']['goal']['kind'])
        self.assertEqual('unknown', packet['workspace']['actors']['microphone_speaker']['identity'])
        public = json.dumps(self.agent.status(), ensure_ascii=False)
        self.assertNotIn('那它在夜里怎么办', public)

    def test_reminder_requires_touch_confirmation_and_uses_real_receipt(self):
        token = self.draft()
        result = self.agent.confirm_action(token)['state']
        self.assertEqual(1, len(self.tasks.creates))
        self.assertEqual('scheduled', result['action']['status'])
        self.assertEqual('scheduled', result['workspace']['goal']['state'])
        self.assertEqual('scheduled', result['action']['task']['state'])
        try:
            self.agent.confirm_action(token)
        except (ValueError, RuntimeError):
            pass
        self.assertEqual(1, len(self.tasks.creates))
        self.assertEqual([], self.kiosk.reasoner.calls)

    def test_modification_revokes_old_confirmation(self):
        old = self.draft()
        new = self.draft('改成五分钟')
        self.assertNotEqual(old, new)
        with self.assertRaises((ValueError, RuntimeError)):
            self.agent.confirm_action(old)
        self.assertEqual([], self.tasks.creates)
        self.agent.confirm_action(new)
        plan = self.tasks.creates[0][1]['proposal']
        self.assertEqual(300, plan['schedule']['delay_seconds'])

    def test_reset_draft_is_not_task_cancellation_and_submitted_requires_explicit_cancel(self):
        self.draft()
        self.agent.reset()
        self.assertEqual([], self.tasks.creates)
        self.assertEqual([], self.tasks.cancels)
        self.assertEqual(0, self.agent.status()['state']['workspace']['history_count'])
        self.agent.confirm_action(self.draft())
        with self.assertRaises(RuntimeError):
            self.agent.reset()
        self.agent.cancel_action()
        self.assertEqual(1, len(self.tasks.cancels))
        self.agent.reset()

    def test_cancelled_draft_cannot_execute(self):
        token = self.draft()
        self.agent.cancel_action()
        with self.assertRaises((ValueError, RuntimeError)):
            self.agent.confirm_action(token)
        self.assertEqual([], self.tasks.creates)

    def test_unsupported_actions_never_claim_to_have_scheduled_a_task(self):
        for message in ('不要十分钟后提醒我休息', '十分钟后提醒我休息并打开相机', '提醒我'):
            result = self.agent.chat(message)
            self.assertEqual('clarify', result['intent'])
            self.assertIn('没有新建', result['answer'])
        self.assertEqual([], self.tasks.creates)
        self.assertEqual([], self.kiosk.reasoner.calls)

    def test_resource_exhaustion_skips_model_without_retry(self):
        now = time.time()
        self.kiosk._perception_context = lambda: {'fusion': {'ready': True, 'items': [
            {'kind': 'resources', 'source': 'control-plane', 'state': 'present',
             'observed_at': now, 'expires_at': now + 10,
             'data': {'cpu_available': 0, 'audio_available': True}}]}}
        with self.assertRaises(RuntimeError):
            self.agent.chat('帮我解释植物的事情')
        self.assertEqual([], self.kiosk.reasoner.calls)
        self.assertFalse(self.agent.budget.status()['busy'])

    def test_expired_object_current_presence_is_host_uncertainty_not_model_echo(self):
        now = time.time()
        self.kiosk._perception_context = lambda: {'objects': [
            {'label': 'cup', 'count': 1, 'source': 'camera.front',
             'observed_at': now - 20, 'expires_at': now - 1}]}
        result = self.agent.chat('刚才那个杯子，现在还在那里吗？')
        self.assertEqual('object-evidence-boundary', result['backend'])
        self.assertIn('这次没有可靠的当前物体线索', result['answer'])
        self.assertNotIn('过期', result['answer'])
        self.assertIn('不能据此确认', result['answer'])
        self.assertEqual([], self.kiosk.reasoner.calls)
        self.assertEqual(0, self.kiosk.visual_calls)
        self.assertIsNone(self.agent.workspace.snapshot()['attention'])
        self.assertEqual([], self.tasks.creates)
        self.assertFalse(result['state']['delivery']['confirmed_terminal'])

    def test_expired_boundary_does_not_intercept_general_use_plans_or_quotes(self):
        from gateway.display.situated import SituatedInteraction
        packet = {'source_states': {'objects': {'state': 'stale'}}, 'observations': {'objects': []}}
        for message in ('杯子一般适合装水吗？', '那它一般适合用来装水吗？',
                        '现在先整理还是阅读？', '刚才那个人现在还在吗？',
                        '你刚才问“杯子现在还在吗”是什么意思？', '杯子现在还在吗？打开相机看看',
                        '提醒任务现在还在吗？'):
            with self.subTest(message=message):
                self.assertIsNone(SituatedInteraction._unconfirmed_object_presence(message, packet))

    def test_expired_boundary_does_not_override_current_or_unavailable_observations(self):
        from gateway.display.situated import SituatedInteraction
        query = '杯子现在还在吗？'
        for state in ('observed_absent', 'available'):
            with self.subTest(state=state):
                self.assertIsNone(SituatedInteraction._unconfirmed_object_presence(query, {
                    'source_states': {'objects': {'state': state}}, 'observations': {'objects': []}}))
        self.assertIsNone(SituatedInteraction._unconfirmed_object_presence(query, {
            'source_states': {'objects': {'state': 'stale'}},
            'observations': {'objects': [{'label': 'cup', 'count': 1}]}}))

    def test_object_boundary_reports_typed_unavailability_without_fabricating_expiry(self):
        from gateway.display.situated import SituatedInteraction
        for state, text in (('stale', '已过期'), ('unknown', '没有可靠'),
                            ('disabled', '未开启'), ('unavailable', '不可用')):
            with self.subTest(state=state):
                reply = SituatedInteraction._unconfirmed_object_presence('杯子现在还在吗？', {
                    'source_states': {'objects': {'state': state}},
                    'observations': {'objects': [], 'people': [{'track_id': 'irrelevant-person'}]}})
                self.assertIn(text, reply['response'])
                if state != 'stale':
                    self.assertNotIn('过期', reply['response'])
                self.assertEqual(0, reply['model_calls'])

    def test_expired_scene_still_allows_real_reasoning_for_general_object_knowledge(self):
        now = time.time()
        self.kiosk._perception_context = lambda: {'objects': [
            {'label': 'cup', 'count': 1, 'source': 'camera.front',
             'observed_at': now - 20, 'expires_at': now - 1}]}
        result = self.agent.chat('杯子一般适合装水吗？')
        self.assertEqual('fake-model', result['backend'])
        self.assertEqual(1, len(self.kiosk.reasoner.calls))
        self.assertEqual(0, self.kiosk.visual_calls)

    def test_delivery_is_not_complete_until_matching_terminal_receipt(self):
        result = self.agent.chat('你好')
        delivery = result['state']['delivery']
        self.assertEqual('queued', delivery['state'])
        self.assertFalse(delivery['confirmed_terminal'])
        self.agent.output_receipt('somebody-else', 'completed', terminal=True)
        self.agent.output_receipt(delivery['job_id'], 'completed', terminal=False)
        self.assertEqual('queued', self.agent.status()['state']['delivery']['state'])
        self.agent.output_receipt(delivery['job_id'], 'completed', terminal=True)
        self.assertTrue(self.agent.status()['state']['delivery']['confirmed_terminal'])

    def test_model_cannot_spoof_execution_receipts_even_with_valid_schema(self):
        for claimed in ('已经设置好十分钟提醒了。', '已经打开相机了。', '我已帮你取消预约了。',
                        '安排好了。', '十分钟后我会提醒你。', 'Done, your alarm is set.', '放心，闹钟搞定啦'):
            self.kiosk.reasoner.reply.update(intent='chat', response=claimed, model_calls=0)
            result = self.agent.chat('你觉得接下来怎么做')
            self.assertNotIn(claimed, result['answer'])
            self.assertIn('还没有执行', result['answer'])
        self.assertEqual([], self.tasks.creates)

    def test_goal_receipt_change_during_model_revokes_old_response(self):
        self.agent.chat('你好')
        self.kiosk.reasoner.callback = lambda: self.agent.workspace.set_goal_state('completed', 'receipt')
        result = self.agent.chat('那然后呢')
        self.assertFalse(result['accepted'])
        self.assertEqual(1, len(self.kiosk.responses))

    def test_status_goal_title_does_not_echo_private_utterance(self):
        sentinel = 'SYNTHETIC_PRIVATE_NAME_123'
        self.agent.chat('我想聊聊' + sentinel)
        self.assertNotIn(sentinel, json.dumps(self.agent.status()))
        self.assertIn(sentinel, self.agent.workspace.dialogue()[0]['content'])

    def test_reminder_draft_replaces_quiz_scope_not_its_receipt(self):
        from interaction_context.situation import TASKS
        self.agent.select(TASKS[0]['id'])
        self.draft()
        state = self.agent.status()['state']
        self.assertIsNone(state['task'])
        self.assertEqual('awaiting_confirmation', state['workspace']['goal']['state'])
        self.agent.chat('5')
        self.assertEqual('awaiting_confirmation', self.agent.status()['state']['workspace']['goal']['state'])

    def test_verified_quiz_status_is_not_confused_with_reminder(self):
        from interaction_context.situation import TASKS
        self.agent.select(TASKS[0]['id'])
        result = self.agent.chat('5')
        self.assertEqual('completed', result['state']['workspace']['goal']['state'])
        self.assertIn('核对通过', self.agent.chat('完成了吗')['answer'])

    def test_old_answer_cannot_mark_replacement_quiz_complete(self):
        from interaction_context.situation import TASKS
        self.agent.select(TASKS[0]['id'])
        original = self.agent.situation.feedback
        def changed(correct, **options):
            self.agent.select(TASKS[1]['id'])
            return original(correct, **options)
        self.agent.situation.feedback = changed
        result = self.agent.chat('5')
        self.assertFalse(result['accepted'])
        state = self.agent.status()['state']
        self.assertEqual(TASKS[1]['id'], state['task']['id'])
        self.assertEqual('active', state['state'])
        self.assertEqual({'correct': 0, 'incorrect': 0, 'recent': []}, state['attempts'])
        self.assertEqual([], self.kiosk.responses)

    def test_old_decline_does_not_change_replacement_task_preference(self):
        from interaction_context.situation import TASKS
        self.agent.select(TASKS[0]['id'])
        original = self.agent.situation.decline
        def changed(**options):
            self.agent.select(TASKS[1]['id'])
            return original(**options)
        self.agent.situation.decline = changed
        result = self.agent.chat('不用提示')
        self.assertFalse(result['accepted'])
        self.assertFalse(self.agent.status()['state']['offers']['declined'])
        self.assertEqual([], self.kiosk.responses)

    def test_output_registration_never_holds_turn_lock_against_control_lock(self):
        from unittest.mock import patch
        from test_display_service import FakeAudio, FakeLLM, FakeMedia, FakeSpeech
        from gateway.display.server import KioskInteraction, DisplayState
        kiosk = KioskInteraction(DisplayState(), FakeLLM(), FakeSpeech(), FakeAudio(), media=FakeMedia())
        kiosk._start_playback = lambda *args, **options: 'synthetic-owned-output'
        registered = []
        def register(job_id, turn_token):
            self.assertFalse(kiosk.turn_manager.lock._is_owned())
            registered.append(job_id)
        kiosk.situated.register_output = register
        token = kiosk.turn_manager.begin('thinking')
        with patch('gateway.display.server.threading.Thread'):
            self.assertEqual('synthetic-owned-output', kiosk._play('测试', turn_token=token))
        self.assertEqual(['synthetic-owned-output'], registered)

    def test_selecting_quiz_revokes_previous_object_focus(self):
        from interaction_context.situation import TASKS
        self.install_scene()
        workspace = self.agent.status()['state']['workspace']
        book = next(row for row in workspace['objects'] if row['label'] == 'book')
        self.agent.attention(book['ref'], workspace['revision'])
        self.agent.select(TASKS[0]['id'])
        self.assertIsNone(self.agent.status()['state']['workspace']['attention'])
        answer = self.agent.chat('这个是什么？')['answer']
        self.assertIn(TASKS[0]['title'], answer)
        self.assertNotIn('book', answer)
        self.assertEqual([], self.kiosk.reasoner.calls)

    def test_ending_quiz_does_not_leave_an_old_goal_or_object_focus(self):
        from interaction_context.situation import TASKS
        self.agent.select(TASKS[0]['id'])
        self.agent.select(None)
        workspace = self.agent.status()['state']['workspace']
        self.assertEqual('继续当前对话', workspace['goal']['title'])
        self.assertIsNone(workspace['attention'])


if __name__ == '__main__':
    unittest.main()
