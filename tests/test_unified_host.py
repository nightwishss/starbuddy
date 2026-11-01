"""Integration against real host/Output with synthetic providers and fake audio."""
import io
import json
import time
import unittest
from types import SimpleNamespace as NS

from interaction_context.core import FusionCore
from interaction_context.host import FusionHost
from orchestration.turn_manager import TurnManager
from output_service.arbiter import OutputArbiter
from perception_context_service.context import SemanticContext
from perception_context_service.service import ContextService
from tests.test_output_arbiter import FakeEngine
from tests.test_unified_interaction import Host


class UnifiedHostTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Host()
        self.display = {'session_id': 'synthetic-session', 'phase': 'idle', 'turn_id': None,
                        'enabled': True, 'output_pending': False, 'quiet': False, 'user_busy': False,
                        'control_epoch': 'a'*32, 'control_revision': 0, 'observed_at': 100.}
        self.output_state = {'ready': True, 'current': None, 'queued': 0}
        self.events = {'epoch': 'events-1', 'events': [], 'revision': 0}
        self.posture = {'state': 'unhealthy', 'enabled': True, 'calibrated': True, 'reasons': [],
                        'source_state': 'fresh', 'generation': 1, 'observed_at': 100.,
                        'expires_at': 103., 'source': 'camera.environment', 'provider': 'posture-policy'}
        self.semantic = {'posture': self.posture}
        def opener(url, **_):
            if url.endswith('/api/interaction/state'): value = self.display
            elif url.endswith('/api/wave/status'): value = self.fixture.live
            elif url.endswith('/api/posture/status'): value = self.posture
            else: raise AssertionError('unexpected endpoint')
            return io.BytesIO(json.dumps(value).encode())
        self.h = FusionHost(FusionCore(clock=self.fixture.clock), clock=self.fixture.clock,
            output=NS(system_status=lambda: self.output_state),
            resources=NS(status=lambda: {'available': {'audio_output': 1, 'cpu_compute': 50,
                                                      'npu_compute': 100}, 'leases': []}),
            events=NS(read=lambda *a, **k: self.events),
            conversation=NS(context=lambda session: {'session_id': session, 'memory': {}}),
            opener=opener, mode='enabled', semantic_provider=lambda: self.semantic)
        self.h.tick(self.semantic)

    def tearDown(self): self.h.close()

    def wave(self, serial=1):
        return self.h.greetings.propose(self.fixture.event(serial, named=False))

    def tick(self):
        self.fixture.now += .01
        self.display['observed_at'] = self.fixture.now
        self.fixture.live['observed_at'] = self.fixture.now
        self.h.tick(self.semantic)

    def test_real_host_reports_idle_and_no_private_state(self):
        value = self.h.snapshot()['interaction_state']
        self.assertEqual(value['phase'], 'idle')
        self.assertNotIn('synthetic-session', json.dumps(value))

    def test_owner_socket_exposes_only_read_only_diagnostics(self):
        service = ContextService(SemanticContext(), fusion=self.h)
        self.assertEqual(service.dispatch({'operation':'fusion.interaction'})['phase'], 'idle')
        with self.assertRaises(ValueError):
            service.dispatch({'operation':'fusion.interaction', 'phase':'idle'})

    def test_posture_and_wave_share_reservation(self):
        first = self.h.issue_guard()
        self.assertIn('guard', first)
        wave = self.wave()
        self.assertEqual(wave['action'], 'greet')
        self.assertFalse(self.h.authorize_guard(first['guard'], 'posture-job')['allowed'])
        self.assertTrue(self.h.authorize_guard(wave['guard'], 'wave-job')['allowed'])

    def test_posture_can_authorize_its_own_output(self):
        proposal = self.h.issue_guard()
        self.output_state['current'] = {'job_id':'posture-job', 'state':'synthesizing'}
        self.assertTrue(self.h.authorize_guard(proposal['guard'], 'posture-job')['allowed'])

    def test_quiet_blocks_posture_even_when_posture_itself_is_valid(self):
        self.display['quiet'] = True; self.tick()
        result = self.h.issue_guard()
        self.assertNotIn('guard', result)
        self.assertEqual(result['decision']['reason'], 'quiet_mode')

    def test_posture_flags_missing_fail_closed(self):
        del self.display['quiet']; self.tick()
        self.assertNotIn('guard', self.h.issue_guard())

    def test_malformed_flags_invalidate_previous_idle_evidence(self):
        self.display['quiet'] = 'false'; self.tick()
        self.assertNotIn('guard', self.h.issue_guard())
        self.assertEqual(self.wave()['action'], 'suppress')

    def test_control_revision_revokes_unseen_fast_user_turn(self):
        proposal = self.wave(); self.display['control_revision'] += 2
        self.assertFalse(self.h.authorize_guard(proposal['guard'], 'job')['allowed'])

    def test_display_restart_revokes_old_ticket(self):
        proposal = self.wave(); self.display['control_epoch'] = 'b'*32
        self.assertFalse(self.h.authorize_guard(proposal['guard'], 'job')['allowed'])

    def test_event_restart_revokes_old_ticket(self):
        proposal = self.wave(); self.events['epoch'] = 'events-2'; self.tick()
        self.assertFalse(self.h.authorize_guard(proposal['guard'], 'job')['allowed'])

    def test_output_receipt_updates_unified_state(self):
        proposal = self.wave()
        self.assertTrue(self.h.authorize_guard(proposal['guard'], 'job')['allowed'])
        self.output_state['current'] = {'job_id':'job', 'state':'playing'}
        self.tick(); self.assertEqual(self.h.snapshot()['interaction_state']['phase'], 'speaking')
        self.output_state['current'] = None
        self.events.update(revision=1, events=[{'revision':1, 'source':'output',
            'topic':'audio.output.completed', 'created_at':self.fixture.now,
            'payload':{'job_id':'job', 'state':'completed'}}])
        self.tick()
        self.assertEqual(self.h.snapshot()['interaction_state']['phase'], 'engaged')
        self.assertEqual(self.h.issue_guard()['decision']['reason'], 'engagement_active')

    def test_read_path_never_calls_slow_providers(self):
        self.h.opener = lambda *a, **k: self.fail('snapshot performed I/O')
        self.assertEqual(self.h.snapshot()['interaction_state']['phase'], 'idle')

    def test_control_generation_is_not_changed_by_robot_playback(self):
        turn = TurnManager(lambda _: None)
        before = turn.snapshot()
        token = turn.begin('listening')
        self.assertEqual(turn.snapshot()['control_revision'], before['control_revision'] + 1)
        turn.transition(token, 'thinking'); turn.attach_output(token, 'job'); turn.output_finished('job')
        self.assertEqual(turn.snapshot()['control_revision'], before['control_revision'] + 1)
        turn.cancel()
        self.assertEqual(turn.snapshot()['control_revision'], before['control_revision'] + 2)

    def run_output(self, change=None):
        h = self.h
        class Engine(FakeEngine):
            def validate_guard(_, guard, job): return h.authorize_guard(guard, job)['allowed']
            def reserve_output(_, owner): return 'synthetic-lease'
            def release_output(_, lease, owner): pass
            def synthesize(engine, text, owner):
                if change: change()
                return super().synthesize(text, owner)
        engine = Engine(); output = OutputArbiter(engine)
        self.h.output = NS(system_status=output.system_status)
        proposal = self.wave(); self.assertEqual(proposal['action'], 'greet')
        output.start()
        try:
            job = output.say(text=proposal['text'], owner='session:wave-greeting', category='notification',
                             guard=proposal['guard'], expires_in_seconds=12)['job_id']
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                state = output.status(job_id=job)['state']
                if state in {'playing', 'cancelled', 'failed'}: break
                time.sleep(.01)
            self.assertEqual(state, 'cancelled' if change else 'playing')
            self.assertEqual(len(engine.started), 0 if change else 1)
        finally: output.close()

    def test_end_to_end_wave_uses_real_guards_and_fake_audio(self): self.run_output()

    def test_end_to_end_user_turn_during_synthesis_cancels(self):
        def change(): self.display['control_revision'] += 1
        self.run_output(change)

    def test_end_to_end_quiet_during_synthesis_cancels(self):
        def change(): self.display['quiet'] = True
        self.run_output(change)

    def test_end_to_end_departure_during_synthesis_cancels(self):
        def change(): self.fixture.live['person_present'] = False
        self.run_output(change)


if __name__ == '__main__': unittest.main(verbosity=2)
