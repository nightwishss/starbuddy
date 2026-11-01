"""Product route→situation→reasoning→delivery tests, synthetic dependencies only."""
import json
from copy import deepcopy
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

from gateway.display.server import Handler, DisplayState
from gateway.display.situated import SituatedInteraction
from interaction_context.situation import TASKS
from orchestration.turn_manager import TurnManager


class Reasoner:
    def __init__(self):
        self.calls = []
        self.callback = None
        self.reply = {'status': 'ok', 'intent': 'help_task', 'response': '先看原来有几个，再接着数。',
                      'requested_capability': None, 'backend': 'fake-model', 'model_calls': 1,
                      'elapsed_ms': 2}

    def status(self):
        return {'available': True, 'backend': 'fake-model', 'hermes_available': False}

    def analyze(self, packet, *, current):
        self.calls.append(packet)
        if self.callback:
            self.callback()
        return dict(self.reply)


class Kiosk:
    def __init__(self):
        self.lock = threading.Lock()
        self.turn_manager = TurnManager(lambda _identifier: None)
        self.state = DisplayState()
        self.quiet = SimpleNamespace(interrupt=lambda: None)
        self.local_llm = None
        self.vision_enabled = False
        self.vision_generation = 1
        self.vision_lock = threading.Lock()
        self.scene_lock = threading.Lock()
        self.context_reads = self.visual_calls = 0
        self.responses = []
        self.reasoner = Reasoner()
        self.situated = SituatedInteraction(self, reasoner=self.reasoner)

    def interaction_status(self):
        return self.turn_manager.snapshot()

    def cancel_current_turn(self):
        return {'cancelled': self.turn_manager.cancel()}

    def _perception_context(self):
        self.context_reads += 1
        return None

    def _refresh_visual_context(self):
        self.visual_calls += 1

    def _detect_latest(self):
        self.visual_calls += 1
        return {'detections': []}

    def _submit_object_context(self, result, counts):
        raise AssertionError('unexpected observation publication')

    @staticmethod
    def _compact_counts(counts):
        return str(counts)

    def _respond(self, text, **kwargs):
        if not self.turn_manager.current(kwargs['turn_token']):
            raise RuntimeError('cancelled')
        self.responses.append(text)
        return {'answer': text, 'playback_id': 'fake-owned-output', 'local_only': True}


class ProductTests(unittest.TestCase):
    def setUp(self):
        self.kiosk = Kiosk()
        self.agent = self.kiosk.situated

    def select(self, index=0):
        self.agent.select(TASKS[index]['id'])

    def test_status_and_select_are_model_and_sensor_free(self):
        self.select()
        value = self.agent.status()
        self.assertEqual(len(value['state']['tasks']), 3)
        self.assertNotIn('answer', value['state']['task'])
        self.assertEqual(self.kiosk.reasoner.calls, [])
        self.assertEqual(self.kiosk.context_reads, 0)

    def test_deictic_task_help_uses_selected_task_and_one_call(self):
        self.select()
        result = self.agent.chat('这个我不会')
        self.assertTrue(result['accepted'])
        packet = self.kiosk.reasoner.calls[0]
        self.assertEqual(packet['situation']['task']['id'], TASKS[0]['id'])
        self.assertEqual(packet['user_input'], '这个我不会')
        self.assertEqual(len(self.kiosk.reasoner.calls), 1)
        self.assertEqual(len(self.kiosk.responses), 1)

    def test_voice_speaker_and_touch_operator_remain_unknown(self):
        self.select()
        self.agent.chat('帮我看看', input_type='voice')
        actors = self.kiosk.reasoner.calls[0]['situation']['actors']
        self.assertFalse(actors['microphone_speaker']['profile_authorized'])
        self.assertFalse(actors['touch_operator']['profile_authorized'])
        self.assertEqual(actors['cross_modal_identity_link'], 'unproven')

    def test_correct_numeric_answer_is_checked_without_model(self):
        self.select()
        result = self.agent.chat('答案是5')
        self.assertIn('答对', result['answer'])
        self.assertEqual(self.kiosk.reasoner.calls, [])
        self.assertEqual(result['state']['state'], 'completed')

    def test_correct_reading_answer_is_supported(self):
        self.select(2)
        self.assertIn('答对', self.agent.chat('为了挡雨')['answer'])
        self.assertEqual(self.kiosk.reasoner.calls, [])

    def test_offer_after_two_wrong_answers_requires_opt_in_and_only_once(self):
        self.select()
        self.agent.offers(True)
        self.assertNotIn('要听听吗', self.agent.chat('1')['answer'])
        self.assertIn('要听听吗', self.agent.chat('2')['answer'])
        self.assertNotIn('要听听吗', self.agent.chat('3')['answer'])

    def test_default_no_unsolicited_offer(self):
        self.select()
        self.agent.chat('1')
        self.assertNotIn('要听听吗', self.agent.chat('2')['answer'])

    def test_explicit_single_scope_task_hint_uses_no_model(self):
        for message in ('这个我不会，能先给一点提示吗？', '给我一个提示', '提示一下',
                        '可以先给我一点提示吗？', '能不能给我一个小提示？'):
            with self.subTest(message=message):
                self.select()
                result = self.agent.chat(message)
                self.assertEqual(TASKS[0]['hint'], result['answer'])
                self.assertEqual('explicit-task-hint', result['backend'])
                self.assertEqual('explicit_request', result['state']['decision']['reason'])
                self.assertEqual('help_task', result['intent'])
                self.assertEqual(0, result['timing_ms']['model_calls'])
                self.assertEqual([], self.kiosk.reasoner.calls)
                self.assertEqual(0, self.kiosk.context_reads)

    def test_explicit_hint_after_decline_does_not_reenable_unsolicited_offers(self):
        self.select()
        self.agent.offers(True)
        self.agent.decline()
        result = self.agent.chat('给我一个提示')
        self.assertEqual(TASKS[0]['hint'], result['answer'])
        self.assertEqual('explicit-task-hint', result['backend'])
        self.assertTrue(result['state']['offers']['declined'])
        self.assertFalse(result['state']['offers']['eligible'])
        self.assertEqual([], self.kiosk.reasoner.calls)

    def test_explicit_refusal_precedes_hint_fast_path(self):
        self.select()
        for message in ('不用提示', '我现在不要提示'):
            with self.subTest(message=message):
                result = self.agent.chat(message)
                self.assertEqual('decline_help', result['intent'])
                self.assertEqual('explicit-control', result['backend'])
                self.assertNotEqual(TASKS[0]['hint'], result['answer'])
                self.assertTrue(result['state']['offers']['declined'])
        self.assertEqual([], self.kiosk.reasoner.calls)

    def test_hint_without_selected_task_clarifies_instead_of_inventing_task(self):
        for message in ('给我一个提示', '提示一下', '这个我不会，能先给一点提示吗？'):
            with self.subTest(message=message):
                result = self.agent.chat(message)
                self.assertEqual('clarify', result['intent'])
                self.assertEqual('missing-task-boundary', result['backend'])
                self.assertIsNone(result['state']['task'])
                self.assertIn('哪一道题', result['answer'])
                self.assertEqual([], self.kiosk.reasoner.calls)

    def test_general_help_why_negated_hint_and_composite_requests_still_use_model(self):
        for message in ('这个我不会', '为什么是这个数', '不是要提示，解释一下为什么',
                        '给我一个提示，然后解释为什么', '提示一下为什么选这个数字',
                        '不要只给提示，详细解释一下'):
            with self.subTest(message=message):
                self.select()
                before = len(self.kiosk.reasoner.calls)
                result = self.agent.chat(message)
                self.assertEqual(before + 1, len(self.kiosk.reasoner.calls))
                self.assertNotEqual('explicit-task-hint', result['backend'])
                self.assertEqual(1, result['timing_ms']['model_calls'])

    def test_decline_suppresses_offer_without_disabling_explicit_help(self):
        self.select()
        self.agent.offers(True)
        self.agent.chat('不用提示')
        self.agent.chat('1')
        self.assertNotIn('要听听吗', self.agent.chat('2')['answer'])
        self.assertTrue(self.agent.chat('现在给个提示')['accepted'])

    def test_missing_task_cannot_invent_task(self):
        result = self.agent.chat('这个我不会')
        self.assertEqual(result['intent'], 'clarify')
        self.assertIn('哪一道题', result['answer'])
        self.assertEqual(self.kiosk.reasoner.calls, [])

    def test_unknown_speaker_does_not_allow_model_to_claim_no_person(self):
        value = self.agent.chat('镜头里有人，你就知道刚才是谁说话了吗？')
        self.assertIn('不能可靠确认', value['answer'])
        self.assertNotIn('没人', value['answer'])
        self.assertEqual(self.kiosk.reasoner.calls, [])

    def test_explicit_long_refusal_is_fast_and_authoritative(self):
        self.select()
        value = self.agent.chat('我现在不用你的帮助，让我自己试试。')
        self.assertEqual(value['intent'], 'decline_help')
        self.assertTrue(value['state']['offers']['declined'])
        self.assertEqual(self.kiosk.reasoner.calls, [])

    def test_cancel_during_model_never_delivers_late_reply(self):
        self.select()
        self.kiosk.reasoner.callback = self.agent.cancel
        result = self.agent.chat('这个我不会')
        self.assertFalse(result['accepted'])
        self.assertEqual(self.kiosk.responses, [])

    def test_switch_task_during_model_discards_old_reply(self):
        self.select()
        self.kiosk.reasoner.callback = lambda: self.agent.select(TASKS[1]['id'])
        self.assertFalse(self.agent.chat('这个我不会')['accepted'])
        self.assertEqual(self.kiosk.responses, [])
        self.assertEqual(self.agent.status()['state']['task']['id'], TASKS[1]['id'])

    def test_no_new_visual_capture_without_explicit_authorized_source(self):
        self.select()
        self.kiosk.reasoner.reply.update(intent='inspect_scene', requested_capability='perception.scene')
        with self.assertRaises(RuntimeError):
            self.agent.chat('看看周围')
        self.assertEqual(self.kiosk.visual_calls, 0)
        self.assertEqual(self.kiosk.responses, [])

    def test_invalid_decision_no_output_and_can_retry(self):
        self.select()
        self.kiosk.reasoner.reply.update(status='invalid', response='')
        with self.assertRaises(RuntimeError): self.agent.chat('这个我不会')
        self.assertEqual(self.kiosk.responses, [])
        self.assertFalse(self.kiosk.lock.locked())

    def test_visual_closed_while_model_thinks_cannot_reopen(self):
        self.select()
        self.kiosk.vision_enabled = True
        self.kiosk.reasoner.callback = lambda: setattr(self.kiosk, 'vision_enabled', False)
        self.kiosk.reasoner.reply.update(intent='inspect_scene', requested_capability='perception.scene')
        with self.assertRaises(RuntimeError): self.agent.chat('画面里有什么')
        self.assertEqual(self.kiosk.visual_calls, 0)

    def test_inferred_decline_cannot_change_user_preference(self):
        self.select()
        self.kiosk.reasoner.reply.update(intent='decline_help', response='我先不帮你。')
        value = self.agent.chat('你觉得呢')
        self.assertEqual(value['intent'], 'clarify')
        self.assertFalse(value['state']['offers']['declined'])

    def test_sources_are_explicitly_last_snapshot_not_live(self):
        self.select(); self.agent.chat('这一步怎么想')
        state = self.agent.status()['state']
        self.assertFalse(state['context_is_live'])
        self.assertIn('context_observed_at', state)


class PerceptionBoundaryTests(unittest.TestCase):
    """Delayed model proposals cannot revive old perceptual facts."""
    def setUp(self):
        self.wall = 1000.0
        self.clock_patch = patch('gateway.display.situated.time.time', side_effect=lambda: self.wall)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.kiosk = Kiosk()
        self.agent = self.kiosk.situated
        self.kiosk.vision_enabled = True
        self.snapshot = self.objects()
        self.reads = 0
        self.kiosk._perception_context = self.read
        self.kiosk.reasoner.reply.update(intent='chat', response='旧现场断言', requested_capability=None)

    def row(self, **values):
        return {'source': 'camera.environment', 'observed_at': self.wall - 1,
                'expires_at': self.wall + 10, 'confidence': .95,
                'provenance': {'provider': 'synthetic-test'}, **values}

    def objects(self):
        return {'objects': [self.row(label='cup', count=1)]}

    def read(self):
        self.reads += 1
        return deepcopy(self.snapshot)

    def assert_discarded(self, result):
        self.assertTrue(result['accepted'])
        self.assertEqual('clarify', result['intent'])
        self.assertNotIn('旧现场断言', result['answer'])
        self.assertIn('不能确认', result['answer'])
        self.assertEqual('uncertain', result['state']['decision']['reason'])
        self.assertEqual(1, len(self.kiosk.reasoner.calls))
        self.assertEqual(0, self.kiosk.visual_calls)
        self.assertFalse(result['state']['context_is_live'])

    def test_expired_input_not_revived_by_same_semantic_fresh_observation(self):
        def changed():
            self.wall += 11
            self.snapshot = self.objects()
        self.kiosk.reasoner.callback = changed
        result = self.agent.chat('画面里有什么')
        self.assert_discarded(result)
        self.assertEqual(2, self.reads)

    def test_expired_selected_task_help_uses_trusted_hint_not_scene_text(self):
        self.agent.select(TASKS[0]['id'])
        self.kiosk.reasoner.reply['intent'] = 'help_task'
        self.kiosk.reasoner.callback = lambda: setattr(self, 'wall', self.wall + 11)
        result = self.agent.chat('这个我不会')
        self.assertEqual(TASKS[0]['hint'], result['answer'])
        self.assertEqual('help_task', result['intent'])
        self.assertEqual('uncertain', result['state']['decision']['reason'])
        self.assertEqual(1, len(self.kiosk.reasoner.calls))

    def test_closed_visual_source_discards_direct_reply_without_reopening(self):
        self.kiosk.reasoner.callback = lambda: setattr(self.kiosk, 'vision_enabled', False)
        self.assert_discarded(self.agent.chat('画面里有什么'))
        self.assertFalse(self.kiosk.vision_enabled)

    def test_new_visual_generation_discards_direct_reply_even_when_enabled(self):
        self.kiosk.reasoner.callback = lambda: setattr(self.kiosk, 'vision_generation', 2)
        self.assert_discarded(self.agent.chat('画面里有什么'))

    def test_changed_count_or_source_discards_direct_reply(self):
        for field, value in (('count', 2), ('label', 'book'), ('source', 'camera.front')):
            with self.subTest(field=field):
                self.snapshot = self.objects()
                self.kiosk.reasoner.callback = lambda field=field, value=value: self.snapshot['objects'][0].update({field: value})
                before = len(self.kiosk.reasoner.calls)
                result = self.agent.chat('画面里有什么')
                self.assertEqual('clarify', result['intent'])
                self.assertNotIn('旧现场断言', result['answer'])
                self.assertEqual(before + 1, len(self.kiosk.reasoner.calls))

    def test_changed_track_or_new_person_discards_direct_reply(self):
        for add_person in (False, True):
            with self.subTest(add_person=add_person):
                self.snapshot = {'people': [self.row(source='camera.front', track_id='anonymous-1')]}
                def changed():
                    if add_person:
                        self.snapshot['people'].append(self.row(source='camera.front', track_id='anonymous-2'))
                    else:
                        self.snapshot['people'][0]['track_id'] = 'anonymous-2'
                self.kiosk.reasoner.callback = changed
                result = self.agent.chat('现在是什么情况')
                self.assertEqual('clarify', result['intent'])
                self.assertNotIn('旧现场断言', result['answer'])

    def test_changed_posture_state_or_affect_discards_direct_reply(self):
        for kind in ('posture', 'affect'):
            with self.subTest(kind=kind):
                person = self.row(source='camera.front', track_id='anonymous-1')
                if kind == 'posture':
                    self.snapshot = {'posture': self.row(source='camera.front', state='healthy', reasons=[])}
                    def changed():
                        self.snapshot['posture'].update(state='unhealthy', reasons=['torso_lean'])
                    message = '现在坐姿如何'
                else:
                    self.snapshot = {'people': [person], 'affect_cues': [
                        self.row(source='camera.front', track_id='anonymous-1', top_label='happy')]}
                    def changed():
                        self.snapshot['affect_cues'][0]['top_label'] = 'sad'
                    message = '我现在是什么表情'
                self.kiosk.reasoner.callback = changed
                result = self.agent.chat(message)
                self.assertEqual('clarify', result['intent'])
                self.assertNotIn('旧现场断言', result['answer'])

    def test_refresh_same_semantics_with_original_still_fresh_is_allowed(self):
        def refresh():
            self.wall += 1
            self.snapshot = self.objects()
        self.kiosk.reasoner.callback = refresh
        result = self.agent.chat('画面里有什么')
        self.assertEqual('旧现场断言', result['answer'])
        self.assertEqual('chat', result['intent'])
        self.assertEqual(2, self.reads)

    def test_independent_front_source_does_not_require_environment_vision_enabled(self):
        self.kiosk.vision_enabled = False
        self.snapshot = {'people': [self.row(source='camera.front', track_id='anonymous-1')]}
        result = self.agent.chat('现在是什么情况')
        self.assertEqual('旧现场断言', result['answer'])
        self.assertEqual(2, self.reads)

    def test_no_perceptual_observations_does_not_reject_task_for_old_robot_state(self):
        self.agent.select(TASKS[0]['id'])
        self.snapshot = {'fusion': {'ready': True, 'items': [
            {'kind': 'resources', 'source': 'resource-pool', 'state': 'present',
             'observed_at': self.wall, 'expires_at': self.wall + .1,
             'data': {'audio_available': True, 'cpu_available': 20}}]}}
        self.kiosk.reasoner.reply.update(intent='help_task', response='先看原来有几个。')
        self.kiosk.reasoner.callback = lambda: setattr(self, 'wall', self.wall + 20)
        result = self.agent.chat('这个我不会')
        self.assertEqual('先看原来有几个。', result['answer'])
        self.assertEqual('help_task', result['intent'])
        self.assertEqual(1, self.reads)

    def test_cancel_during_metadata_recheck_wins_over_safe_fallback(self):
        def interrupted():
            value = self.read()
            if self.reads == 2:
                self.agent.cancel()
            return value
        self.kiosk._perception_context = interrupted
        result = self.agent.chat('画面里有什么')
        self.assertFalse(result['accepted'])
        self.assertEqual([], self.kiosk.responses)

    def test_metadata_read_failure_drops_reply_without_exception_content(self):
        def unavailable():
            if self.reads:
                raise RuntimeError('SYNTHETIC_PRIVATE_FAILURE')
            return self.read()
        self.kiosk._perception_context = unavailable
        result = self.agent.chat('画面里有什么')
        self.assert_discarded(result)
        self.assertNotIn('SYNTHETIC_PRIVATE_FAILURE', json.dumps(result, ensure_ascii=False))


class VoiceProductTests(unittest.TestCase):
    def create(self):
        from test_display_service import FakeAudio, FakeLLM, FakeMedia, FakeSpeech
        from gateway.display.server import KioskInteraction
        class Audio(FakeAudio):
            def __init__(self):
                super().__init__(); self.captures = 0
            def capture_turn(self, *args, **kwargs):
                self.captures += 1
                return super().capture_turn(*args, **kwargs)
        audio = Audio(); media = FakeMedia()
        kiosk = KioskInteraction(DisplayState(), FakeLLM(), FakeSpeech(), audio, media=media)
        kiosk.situated.reasoner = Reasoner()
        kiosk.situated.select(TASKS[0]['id'])
        return kiosk, audio, media

    def test_revoked_ticket_does_not_start_microphone_or_listening(self):
        kiosk, audio, _ = self.create()
        ticket = kiosk.situated.situation.begin('voice')
        kiosk.situated.cancel()
        result = kiosk.voice_turn(agent_ticket=ticket)
        self.assertFalse(result['accepted'])
        self.assertEqual(audio.captures, 0)
        self.assertEqual(kiosk.turn_manager.snapshot()['phase'], 'idle')
        self.assertFalse(kiosk.voice_lock.locked())

    def test_cancelled_situated_voice_is_not_overwritten_as_accepted(self):
        kiosk, audio, media = self.create()
        kiosk.situated.reasoner.callback = kiosk.situated.cancel
        result = kiosk.situated.voice()
        self.assertFalse(result['accepted'])
        self.assertNotIn('transcript', result)
        self.assertEqual(audio.captures, 1)
        self.assertEqual(audio.calls, [])
        self.assertTrue(media.released)

    def test_cancel_during_output_drain_does_not_start_capture(self):
        from unittest.mock import patch
        kiosk, audio, _ = self.create()
        busy = iter((True, False, False))
        kiosk._output_busy_for_capture = lambda: next(busy)
        with patch('gateway.display.server.time.sleep', side_effect=lambda _: kiosk.situated.cancel()):
            result = kiosk.situated.voice()
        self.assertFalse(result['accepted'])
        self.assertEqual(audio.captures, 0)
        self.assertFalse(kiosk.voice_lock.locked())
        self.assertEqual(kiosk.turn_manager.snapshot()['phase'], 'idle')

    def test_cancel_during_listening_update_does_not_start_capture(self):
        kiosk, audio, _ = self.create()
        update = kiosk.state.update
        def interrupted(values):
            update(values)
            if values.get('phase') == 'listening':
                kiosk.situated.cancel()
        kiosk.state.update = interrupted
        result = kiosk.situated.voice()
        self.assertFalse(result['accepted'])
        self.assertEqual(audio.captures, 0)
        self.assertFalse(kiosk.voice_lock.locked())


class FixedDeliveryTests(unittest.TestCase):
    # The production Kiosk/TurnManager→Output route, no native playback.
    create = VoiceProductTests.create
    def configured(self, *, interrupt_import=False, fail_enqueue=False):
        kiosk, audio, _ = self.create()
        class Output:
            def __init__(self): self.calls = []
            def play_media(self, media_id, owner, **kwargs):
                self.calls.append(('play_media', media_id, owner, kwargs))
                if fail_enqueue: raise RuntimeError('unknown enqueue outcome')
                return {'job_id': 'output_abcdefghijklmnopqrstuvwx'}
            def say(self, *args, **kwargs):
                self.calls.append(('say', args, kwargs))
                return {'job_id': 'output_abcdefghijklmnopqrstuvwx'}
            def cancel(self, identifier): self.calls.append(('cancel', identifier))
        class Prepared:
            def __init__(self): self.owned = False; self.released = []; self.transferred = []
            def status(self): return {'ready': True}
            def prepare(self, text, *, speed=1.0):
                if speed != 1.0: return None
                self.owned = True
                if interrupt_import: kiosk.situated.cancel()
                return 'generated_abcdefghijklmnopqrstuvwx'
            def handoff(self, identifier):
                if not self.owned: return False
                self.owned = False; self.transferred.append(identifier); return True
            def release(self, identifier):
                if self.owned: self.released.append(identifier); self.owned = False
        kiosk.output = Output()
        kiosk.situated.fixed_audio = Prepared()
        kiosk._watch_playback = lambda *args: None
        return kiosk, audio

    def test_fixed_hint_uses_same_output_and_turn_ownership_without_tts_or_model(self):
        kiosk, audio = self.configured()
        value = kiosk.situated.chat('给我一个提示')
        self.assertTrue(value['fixed_audio_used'])
        self.assertEqual(value['timing_ms']['model_calls'], 0)
        self.assertEqual(kiosk.output.calls[0][:3], ('play_media', 'generated_abcdefghijklmnopqrstuvwx', 'session:display-kiosk'))
        self.assertEqual(kiosk.turn_manager.snapshot()['output_id'], value['playback_id'])
        self.assertEqual(audio.calls, [])
        self.assertFalse(kiosk.situated.fixed_audio.owned)
        self.assertEqual(kiosk.situated.fixed_audio.released, [])
        kiosk.situated.cancel()
        self.assertIn(('cancel', value['playback_id']), kiosk.output.calls)

    def test_cancel_during_fixed_media_import_releases_only_owned_clip_without_output(self):
        kiosk, audio = self.configured(interrupt_import=True)
        value = kiosk.situated.chat('给我一个提示')
        self.assertFalse(value['accepted'])
        self.assertEqual(kiosk.output.calls, [])
        self.assertEqual(kiosk.situated.fixed_audio.released, ['generated_abcdefghijklmnopqrstuvwx'])
        self.assertEqual(audio.calls, [])

    def test_slow_preferences_never_use_normal_speed_fixed_asset(self):
        kiosk, audio = self.configured()
        kiosk.preferences = SimpleNamespace(snapshot=lambda: {'values': {'speed': .7}})
        value = kiosk.situated.chat('给我一个提示')
        self.assertFalse(value['fixed_audio_used'])
        self.assertEqual(kiosk.output.calls[0][0], 'say')
        self.assertEqual(kiosk.output.calls[0][2]['speed'], .7)
        self.assertEqual(audio.calls, [])

    def test_unknown_enqueue_does_not_retry_or_release_transferred_clip(self):
        kiosk, audio = self.configured(fail_enqueue=True)
        with self.assertRaises(RuntimeError): kiosk.situated.chat('给我一个提示')
        self.assertEqual(len(kiosk.output.calls), 1)
        self.assertFalse(kiosk.situated.fixed_audio.owned)
        self.assertEqual(kiosk.situated.fixed_audio.released, [])
        self.assertEqual(kiosk.turn_manager.snapshot()['phase'], 'idle')
        self.assertEqual(audio.calls, [])


class RoutesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.server.display_state = DisplayState()
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = 'http://127.0.0.1:%s' % cls.server.server_port

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def setUp(self):
        self.kiosk = Kiosk(); self.server.kiosk_interaction = self.kiosk

    def call(self, route, data=None, headers=None):
        body = None if data is None else json.dumps(data).encode()
        request = urllib.request.Request(self.url + route, data=body,
                    headers={'Content-Type': 'application/json', **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as response:
            return response.code, response.read()

    def test_page_assets_and_state_are_real_routes_without_effects(self):
        for route in ('/interaction', '/interaction.css', '/interaction.js', '/api/agent/state'):
            status, body = self.call(route)
            self.assertEqual(status, 200, route); self.assertTrue(body)
        self.assertEqual(self.kiosk.reasoner.calls, [])
        self.assertEqual(self.kiosk.context_reads, 1)  # bounded anonymous snapshot, not new capture
        self.assertEqual(self.kiosk.visual_calls, 0)

    def test_touch_api_to_model_packet_to_response(self):
        self.assertEqual(self.call('/api/agent/task', {'task_id': TASKS[0]['id']})[0], 200)
        code, raw = self.call('/api/agent/chat', {'message': '这个不会'})
        self.assertEqual(code, 200)
        self.assertTrue(json.loads(raw)['accepted'])
        self.assertEqual(len(self.kiosk.reasoner.calls), 1)

    def test_cross_origin_control_refused(self):
        code, _ = self.call('/api/agent/task', {'task_id': TASKS[0]['id']}, {'Origin': 'https://evil.invalid'})
        self.assertEqual(code, 403)
        self.assertIsNone(self.kiosk.situated.situation.snapshot()['task'])

    def test_extra_authority_fields_refused(self):
        code, _ = self.call('/api/agent/chat', {'message': 'hi', 'profile_id': 'other'})
        self.assertEqual(code, 400)

    def test_unknown_task_and_nonboolean_offer_rejected(self):
        for route, payload in (('/api/agent/task', {'task_id': 'invented'}),
                               ('/api/agent/offers', {'enabled': 1})):
            self.assertEqual(self.call(route, payload)[0], 400)

    def test_reminder_routes_require_confirmation_and_reject_extra_authority(self):
        from gateway.display.situated_actions import ReminderActionBroker
        from test_multimodal_story import TaskReceipts
        tasks = TaskReceipts()
        self.kiosk.situated.actions = ReminderActionBroker(tasks)
        code, raw = self.call('/api/agent/chat', {'message': '十分钟后提醒我喝水'})
        self.assertEqual(200, code)
        token = json.loads(raw)['state']['action']['draft']['token']
        self.assertEqual([], tasks.creates)
        self.assertEqual(400, self.call('/api/agent/confirm', {'token': token, 'profile_id': 'forged'})[0])
        self.assertEqual(403, self.call('/api/agent/confirm', {'token': token}, {'Origin': 'https://evil.invalid'})[0])
        code, raw = self.call('/api/agent/confirm', {'token': token})
        self.assertEqual(200, code)
        self.assertEqual('scheduled', json.loads(raw)['state']['action']['status'])
        self.assertEqual(409, self.call('/api/agent/confirm', {'token': token})[0])
        self.assertEqual(1, len(tasks.creates))

    def test_status_get_reconciles_unknown_without_dispatching_pending_cancel(self):
        from gateway.display.situated_actions import ReminderActionBroker
        from test_multimodal_story import TaskReceipts
        class LostAck(TaskReceipts):
            def create(self, utterance, **values):
                super().create(utterance, **values)
                raise TimeoutError('synthetic lost acknowledgement')
        tasks = LostAck()
        self.kiosk.situated.actions = ReminderActionBroker(tasks)
        _, raw = self.call('/api/agent/chat', {'message': '十分钟后提醒我休息'})
        token = json.loads(raw)['state']['action']['draft']['token']
        self.assertEqual(409, self.call('/api/agent/confirm', {'token': token})[0])
        self.assertEqual(200, self.call('/api/agent/action-cancel', {})[0])
        code, raw = self.call('/api/agent/state')
        self.assertEqual(200, code)
        self.assertEqual('cancel_requires_explicit_retry', json.loads(raw)['state']['action']['reason'])
        self.assertEqual([], tasks.cancels)
        self.assertEqual(200, self.call('/api/agent/action-cancel', {})[0])
        self.assertEqual(1, len(tasks.cancels))


if __name__ == '__main__':
    unittest.main()
