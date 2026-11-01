"""Policy regression with fake camera, identity, models and audio only."""
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from gateway.display.server import KioskInteraction, DisplayState
from gateway.display.wave import WaveController
from tests.test_display_service import (FakeLLM, FakeSpeech, FakeAudio, FakePreview,
    FakeVision, FakeMedia, FakeConversation, FakePerceptionContext)
from tests.test_face_identity_service import FakeVault, FakeEvents, FakeOutput, candidate
from face_identity_service.engine import IdentityService
from interaction_context.core import FusionCore, Access
from interaction_context.host import FusionHost
from perception_context_service.context import SemanticContext
from perception_context_service.service import ContextService


class SilentVisualTests(unittest.TestCase):
    def interaction(self):
        state, llm, audio, context, conversation = (DisplayState(), FakeLLM(), FakeAudio(),
                                                   FakePerceptionContext(), FakeConversation())
        app = KioskInteraction(state, llm, FakeSpeech(), audio, FakePreview(), FakeVision(),
                               FakeMedia(), perception_context=context, conversation=conversation)
        app.vision_enabled = True
        app.wave = Mock()
        return app, state, llm, audio, context, conversation

    def observe(self, app, reason, announce):
        app.scene_lock.acquire()
        app._observation_worker(reason, announce, app.vision_generation)

    def test_initial_manual_periodic_observation_updates_context_without_speech(self):
        for reason, announce in [('initial', True), ('manual', True), ('monitor', False)]:
            with self.subTest(reason=reason):
                app, state, llm, audio, context, conversation = self.interaction()
                before = state.snapshot()
                self.observe(app, reason, announce)
                self.assertEqual(context.observations[-1]['data']['counts'], {'cup':1,'person':1})
                self.assertEqual(context.observations[-1]['ttl_seconds'], 30)
                self.assertEqual(app.vision_status()['response_policy'], 'context_only')
                self.assertEqual(audio.calls, []);self.assertEqual(llm.messages, [])
                self.assertEqual(state.snapshot(), before)
                self.assertEqual(conversation.remembered, [])
                self.assertFalse(app.scene_lock.locked())

    def test_arrival_still_emits_context_event_without_speech(self):
        app, state, llm, audio, context, _ = self.interaction()
        app._update_presence(False)
        self.observe(app, 'monitor', False);self.observe(app, 'monitor', False)
        app.wave.arrival.assert_called_once_with()
        self.assertEqual(len(context.observations), 2)
        self.assertEqual(audio.calls, []);self.assertEqual(llm.messages, [])

    def test_observation_error_is_status_only_not_conversation_interruption(self):
        app, state, llm, audio, _, _ = self.interaction()
        before = state.snapshot()
        app._detect_latest = Mock(side_effect=RuntimeError('fake unavailable'))
        self.observe(app, 'manual', True)
        self.assertEqual(app.vision_status()['error'], 'vision_unavailable')
        self.assertEqual(state.snapshot(), before);self.assertEqual(audio.calls, [])

    def test_stopped_or_old_generation_does_not_update_context(self):
        app, _, _, audio, context, _ = self.interaction()
        app.vision_enabled = False
        self.observe(app, 'initial', True)
        self.assertEqual(context.observations, []);self.assertEqual(audio.calls, [])

    def test_legacy_visual_response_blocked_but_explicit_response_unchanged(self):
        app, _, _, _, _, _ = self.interaction()
        app._play = Mock(return_value='fake-job')
        self.assertTrue(app._respond('person', primary='person', secondary='', producer='vision')['suppressed'])
        app._play.assert_not_called()
        result = app._respond('hello', primary='hello', secondary='', producer='conversation')
        self.assertEqual(result['playback_id'], 'fake-job');app._play.assert_called_once()

    def test_passive_arrival_cannot_speak_even_if_fusion_returns_greet(self):
        app = NS(output=NS(socket_path='fake'))
        controller = WaveController(app)
        with patch('gateway.display.wave.PerceptionContextClient') as context, \
                patch('gateway.display.wave.OutputClient') as output:
            context.return_value.call.return_value = {'action':'greet','reason':'legacy-greet'}
            controller.arrival()
            self.assertEqual(context.return_value.call.call_args.kwargs['event']['kind'], 'arrival')
            output.assert_not_called()

    def test_passive_context_does_not_require_output_service(self):
        controller = WaveController(NS(output=None))
        with patch('gateway.display.wave.PerceptionContextClient') as context:
            context.return_value.call.return_value = {'action':'suppress','reason':'context_only'}
            controller.arrival();context.return_value.call.assert_called_once()

    def test_wave_still_delivers_guarded_speech(self):
        app = NS(output=NS(socket_path='fake'), preferences=Mock(), playback_lock=threading.RLock(),
                 active_playback_id=None)
        app.preferences.snapshot.return_value = {'values':{'speed':1.0}}
        controller = WaveController(app)
        with patch('gateway.display.wave.PerceptionContextClient') as context, \
                patch('gateway.display.wave.OutputClient') as output, \
                patch('gateway.display.wave.threading.Thread'):
            context.return_value.call.return_value = {'action':'greet','reason':'greet_named',
                'text':'你好April','expires_in_seconds':12,'guard':{'kind':'greeting','token':'a'*32}}
            output.return_value.say.return_value = {'job_id':'fake-job'}
            controller._submit({'kind':'wave','event_id':'b'*32,'generation':1})
            output.return_value.say.assert_called_once()
            self.assertEqual(output.return_value.say.call_args.kwargs['guard']['kind'], 'greeting')

    def test_recognition_transition_and_legacy_call_are_silent(self):
        vault, events, output = FakeVault(), FakeEvents(), FakeOutput()
        service = IdentityService(object(), camera=object(), vault=vault, events=events, output=output)
        for _ in range(3):service._recognize([candidate()])
        self.assertEqual(service.status()['observation']['status'], 'known')
        self.assertEqual(len(events.values),1)
        service._say('legacy greeting','recognition:fake',cooldown=0)
        self.assertEqual(output.values, [])

    def test_detected_objects_reach_actual_fusion_with_expiry(self):
        now = [100.0]
        core = FusionCore(clock=lambda:now[0])
        host = FusionHost(core, output=Mock(), events=Mock(), resources=Mock(), conversation=Mock(),
                          clock=lambda:now[0])
        context = SemanticContext(clock=lambda:now[0])
        service = ContextService(context, fusion=host)
        result = service.dispatch({'operation':'context.observe','observation':{
            'provider':'yolox-qnn','kind':'objects','source':'camera.environment',
            'observed_at':100.0,'ttl_seconds':30,'confidence':.9,'inference':True,
            'data':{'counts':{'person':1,'cup':1},'confidences':{'person':.9,'cup':.8},'model':'fake'}}})
        self.assertTrue(result['accepted'])
        items = core.snapshot(Access())['items']
        self.assertEqual(next(v for v in items if v['kind']=='objects')['data']['counts'], {'person':1,'cup':1})
        host.output.say.assert_not_called()
        now[0] = 131.0
        self.assertEqual(next(v for v in core.snapshot()['items'] if v['kind']=='objects')['data'], {})


if __name__ == '__main__':unittest.main()
