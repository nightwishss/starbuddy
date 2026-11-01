"""Pure semantic replay: no camera/model/audio/network or private state."""
import hashlib
import json
import threading
import unittest
from copy import deepcopy

from interaction_context.core import Access, FusionCore
from interaction_context.engagement import InteractionArbiter
from interaction_context.greeting import GreetingFusion


class Host:
    def __init__(self):
        self.now = 100.
        self.clock = lambda: self.now
        self.core = FusionCore(clock=self.clock)
        self.lock = threading.RLock()
        self.active_session = 'synthetic-session'
        self.closed = False
        self.semantic_provider = None
        self.last_semantic = {}
        self.interaction = InteractionArbiter(self.clock)
        self.control_revision = 0
        self.live = {'enabled': True, 'arrival_enabled': False, 'generation': 1,
                     'quiet': False, 'user_busy': False, 'turn_active': False,
                     'output_job': None, 'person_present': True, 'body_epoch': 0,
                     'observed_at': self.now}
        self.flags = {'phase': 'idle', 'turn_id': None, 'quiet': False, 'user_busy': False}
        self.out = {'busy': False, 'queued': 0, 'state': 'idle', 'job_id': None}
        self.track, self.label = 'face_0', 'TestUser'
        self.fill()
        self.greetings = GreetingFusion(self)

    def _interaction_boundary(self):
        return (self.core.epoch, self.active_session, self.control_revision)

    def fill(self):
        self.live['observed_at'] = self.now
        for kind, source, data, subject, session in (
            ('presence', 'camera.front', {'count': 1, 'track_ids': [self.track]}, None, None),
            ('identity', 'camera.front', {'label': self.label, 'consent': 'enrolled_local',
                                         'track_id': self.track}, self.track, None),
            ('interaction', 'conversation', self.flags, None, self.active_session),
            ('output', 'output-arbiter', self.out, None, None),
            ('resources', 'control-plane', {'audio_available': True, 'cpu_available': 50,
                                          'npu_available': 100}, None, None)):
            self.core.observe(provider='synthetic', kind=kind, source=source, data=data,
                              subject=subject, session_id=session, observed_at=self.now, ttl_seconds=3)

    def _ingest_semantic(self, _): pass
    def _poll_interaction(self): pass
    def _poll_output(self): pass
    def _poll_resources(self): pass
    def _read_json(self, _): return dict(self.live)
    def snapshot(self):
        return self.core.snapshot(Access(session_id=self.active_session, identity=True))

    def update(self):
        return self.interaction.update(self.snapshot(), self._interaction_boundary(), self.live)

    def event(self, serial=1, named=True):
        return {'kind': 'wave', 'event_id': f'{serial:032x}', 'source': 'camera.front',
                'observed_at': self.now, 'confidence': .98, 'track_id': self.track if named else None,
                'identity_key': hashlib.sha256(self.label.encode()).hexdigest() if named else None,
                'generation': 1, 'body_epoch': self.live['body_epoch']}

    def own_output(self, state='synthesizing'):
        self.now += .01
        self.flags['phase'] = 'acting'
        self.live['output_job'] = 'owned-job'
        self.out.update(busy=True, job_id='owned-job', state=state)
        self.fill()

    def idle(self, seconds=21):
        self.now += seconds
        self.flags['phase'] = 'idle'
        self.live['output_job'] = None
        self.out.update(busy=False, job_id=None, state='idle')
        self.fill()
        self.update()

    def delivered(self):
        proposal = self.greetings.propose(self.event())
        self.own_output()
        assert self.greetings.authorize(proposal['guard'], 'owned-job')['allowed']
        self.interaction.receipt('owned-job', 'playing')
        self.interaction.receipt('owned-job', 'completed')
        self.idle()
        return proposal


class UnifiedInteractionTests(unittest.TestCase):
    def setUp(self):
        self.h = Host()
        self.a = self.h.interaction
        self.h.update()

    def reserve(self, kind='posture', token='ticket'):
        return self.a.reserve(kind, token, self.h.now + 12, self.h.snapshot(),
                              subject=self.a.subject(self.h.event()) if kind == 'wave' else None)

    def test_initial_state_idle_no_microphone(self):
        value = self.a.read()
        self.assertEqual(value['phase'], 'idle')
        self.assertFalse(value['microphone_started'])

    def test_listening_thinking_and_user_busy_block_ambient(self):
        for change in ({'phase': 'listening'}, {'phase': 'thinking'}, {'user_busy': True},
                       {'turn_id': 'new-user-turn'}):
            h = Host(); h.flags.update(change); h.now += .1; h.fill(); h.update()
            for kind in ('wave', 'posture'):
                result = h.interaction.reserve(kind, 't', 112, h.snapshot())
                self.assertFalse(result['allowed'])
                self.assertEqual(result['reason'], 'user_turn_active')

    def test_quiet_blocks_both_paths(self):
        self.h.flags['quiet'] = True; self.h.now += .1; self.h.fill(); self.h.update()
        self.assertEqual(self.a.read()['phase'], 'quiet')
        self.assertEqual(self.reserve()['reason'], 'quiet_mode')
        self.assertEqual(self.h.greetings.propose(self.h.event())['action'], 'suppress')

    def test_missing_flags_fail_closed(self):
        self.h.flags.pop('quiet'); self.h.now += .1; self.h.fill(); self.h.update()
        self.assertEqual(self.reserve()['reason'], 'interaction_flags_unknown')

    def test_stale_state_never_advertises_idle(self):
        self.h.now += 4
        self.assertEqual(self.a.read()['phase'], 'unavailable')
        self.assertFalse(self.reserve()['allowed'])

    def test_output_unknown_and_terminal_cleanup_still_busy(self):
        self.h.out.update(busy=True, state='cancelled', job_id='draining')
        self.h.now += .1; self.h.fill(); self.h.update()
        self.assertEqual(self.reserve()['reason'], 'output_busy')
        self.assertEqual(self.a.read()['phase'], 'acting')

    def test_passive_observations_never_reserve_or_speak(self):
        for kind in ('arrival', 'identity', 'objects', 'affect'):
            self.assertEqual(self.reserve(kind)['reason'], 'context_only')
        self.assertIsNone(self.a.pending)
        self.assertIsNone(self.a.episode)

    def test_wave_reservation_attends_but_does_not_count_as_delivered(self):
        self.assertTrue(self.reserve('wave')['allowed']); self.h.update()
        self.assertEqual(self.a.read()['phase'], 'attending')
        self.assertIsNone(self.a.episode)
        self.assertNotIn('delivered', self.a.read()['counts'])

    def test_single_reservation_prevents_two_ambient_jobs(self):
        self.assertTrue(self.reserve('wave')['allowed'])
        self.assertEqual(self.reserve('posture', 'p')['reason'], 'ambient_reserved')
        self.assertEqual(self.reserve('wave', 'w')['reason'], 'ambient_reserved')

    def test_wave_supersedes_only_unclaimed_posture(self):
        self.reserve('posture', 'p')
        self.assertTrue(self.reserve('wave', 'w')['allowed'])
        self.assertFalse(self.a.claim('p', 'posture-job'))

    def test_wave_never_preempts_claimed_posture(self):
        self.reserve('posture', 'p'); self.assertTrue(self.a.claim('p', 'posture-job'))
        self.assertFalse(self.reserve('wave', 'w')['allowed'])

    def test_claimed_job_exception_is_narrow(self):
        p = self.h.greetings.propose(self.h.event())
        self.h.own_output()
        self.assertTrue(self.h.greetings.authorize(p['guard'], 'owned-job')['allowed'])
        self.assertFalse(self.h.greetings.authorize(p['guard'], 'other-job')['allowed'])

    def test_poll_between_output_dequeue_and_guard_rpc_keeps_handoff(self):
        p = self.h.greetings.propose(self.h.event())
        self.h.own_output(); self.h.update()
        self.assertTrue(self.h.greetings.authorize(p['guard'], 'owned-job')['allowed'])

    def test_handoff_never_exempts_a_different_current_job(self):
        p = self.h.greetings.propose(self.h.event())
        self.h.own_output(); self.h.update()
        self.assertFalse(self.h.greetings.authorize(p['guard'], 'different-job')['allowed'])

    def test_user_turn_permanently_revokes_pending_greeting(self):
        p = self.h.greetings.propose(self.h.event())
        self.h.flags.update(phase='listening', turn_id='user-turn')
        self.h.now += .1; self.h.fill(); self.h.update()
        self.h.flags.update(phase='idle', turn_id=None)
        self.h.now += .1; self.h.fill(); self.h.update()
        self.assertFalse(self.h.greetings.authorize(p['guard'], 'owned-job')['allowed'])

    def test_session_switch_revokes_even_if_switch_back(self):
        p = self.h.greetings.propose(self.h.event())
        old = self.h.active_session
        self.h.active_session = 'different-session'; self.h.fill(); self.h.update()
        self.h.active_session = old; self.h.update()
        self.assertFalse(self.h.greetings.authorize(p['guard'], 'owned-job')['allowed'])

    def test_fast_user_turn_between_polls_revokes(self):
        p = self.h.greetings.propose(self.h.event())
        self.h.control_revision += 2  # begin + cancel; phase is already idle again
        self.h.update()
        self.assertFalse(self.h.greetings.authorize(p['guard'], 'owned-job')['allowed'])

    def test_unrelated_receipts_do_not_create_episode(self):
        self.reserve('wave'); self.a.receipt('other-job', 'completed')
        self.assertIsNone(self.a.episode)

    def test_failed_synthesis_never_creates_greeted_episode(self):
        p = self.h.greetings.propose(self.h.event()); self.h.own_output()
        self.assertTrue(self.h.greetings.authorize(p['guard'], 'owned-job')['allowed'])
        self.a.receipt('owned-job', 'failed'); self.h.idle()
        second = self.h.greetings.propose(self.h.event(2))
        self.assertEqual(second['response_style'], 'greeting')

    def test_playback_receipt_starts_episode_and_repeat_is_short(self):
        self.h.delivered()
        self.assertEqual(self.a.read()['phase'], 'engaged')
        second = self.h.greetings.propose(self.h.event(2))
        self.assertEqual(second['text'], '我在呢，你说。')
        self.assertEqual(second['response_style'], 'continuation')
        self.assertEqual(self.a.read()['counts']['delivered'], 1)

    def test_completed_receipt_recovers_missed_start_without_double_count(self):
        self.reserve('wave'); self.a.claim('ticket', 'job')
        self.a.receipt('job', 'completed'); self.a.receipt('job', 'completed')
        self.assertEqual(self.a.read()['counts']['delivered'], 1)

    def test_engaged_episode_defers_posture_but_not_explicit_wave(self):
        self.h.delivered()
        self.assertEqual(self.reserve()['reason'], 'engagement_active')
        self.assertTrue(self.reserve('wave')['allowed'])

    def test_episode_expires(self):
        self.h.delivered(); self.h.idle(61)
        self.assertEqual(self.a.read()['phase'], 'idle')
        self.assertEqual(self.h.greetings.propose(self.h.event(2))['response_style'], 'greeting')

    def test_body_change_departure_and_disabled_clear_episode(self):
        for change in ({'body_epoch': 2}, {'person_present': False}, {'enabled': False}):
            h = Host(); h.delivered(); h.live.update(change); h.update()
            self.assertIsNone(h.interaction.episode)

    def test_bound_face_change_clears_episode(self):
        self.h.delivered(); self.h.track = 'different-track'; self.h.now += .1
        self.h.fill(); self.h.update(); self.assertIsNone(self.a.episode)

    def test_unnamed_wave_does_not_require_visible_face(self):
        p = self.h.greetings.propose(self.h.event(named=False))
        self.assertEqual(p['action'], 'greet'); self.assertEqual(p['reason'], 'greet_generic')
        self.assertTrue(self.h.greetings.authorize(p['guard'], 'job')['allowed'])

    def test_identity_expiry_cannot_be_rescued_by_episode(self):
        self.h.delivered(); p = self.h.greetings.propose(self.h.event(2))
        for record in self.h.core.records.values():
            if record['kind'] == 'identity': record['expires_at'] = self.h.now - 1
        self.assertFalse(self.h.greetings.authorize(p['guard'], 'job')['allowed'])

    def test_state_observations_dont_extend_episode(self):
        self.h.delivered(); until = self.a.episode['until']
        for _ in range(5): self.h.idle(1)
        self.assertEqual(self.a.episode['until'], until)

    def test_raw_data_names_ids_and_tokens_absent_from_diagnostics(self):
        self.h.delivered()
        text = json.dumps(self.a.read())
        for forbidden in ('TestUser', 'face_0', 'synthetic-session', 'owned-job', 'identity_key'):
            self.assertNotIn(forbidden, text)

    def test_diagnostics_bounded(self):
        for _ in range(500): self.reserve('objects')
        self.assertLessEqual(len(self.a.read()['decisions']), 32)

    def test_read_never_mutates_or_grants_authority(self):
        before = deepcopy(self.a.pending)
        for _ in range(100): self.a.read()
        self.assertEqual(self.a.pending, before)

    def test_core_rejects_non_boolean_flags(self):
        with self.assertRaises(ValueError):
            self.h.core.observe(provider='bad', kind='interaction', source='conversation',
                                observed_at=100, ttl_seconds=3, data={'phase':'idle','quiet':'false'})

    def test_concurrent_proposals_have_one_winner(self):
        results = []
        workers = [threading.Thread(target=lambda i=i: results.append(self.reserve('wave', str(i))))
                   for i in range(8)]
        for worker in workers: worker.start()
        for worker in workers: worker.join()
        self.assertEqual(sum(r['allowed'] for r in results), 1)

    def test_close_revokes(self):
        self.reserve('wave'); self.a.close()
        self.assertFalse(self.a.claim('ticket', 'job'))


if __name__ == '__main__': unittest.main(verbosity=2)
