"""Pure synthetic-clock workspace contracts; no I/O, service, model or media."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import unittest

from interaction_context.workspace import Workspace


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000.0
        self.workspace = Workspace(clock=lambda: self.now)

    def object(self, **values):
        return {'source': 'camera.environment', 'label': 'cup', 'count': 1,
                'observed_at': self.now - 1, 'expires_at': self.now + 10, **values}

    def person(self, **values):
        return {'source': 'camera.front', 'track_id': 'anonymous-track-1',
                'observed_at': self.now - 1, 'expires_at': self.now + 10, **values}

    def packet(self, *, objects=None, people=None, object_state='available', people_state='available'):
        return {'observations': {'objects': [] if objects is None else objects,
                                 'people': [] if people is None else people},
                'source_states': {'objects': {'state': object_state}, 'people': {'state': people_state}}}

    def ingest(self, **values):
        return self.workspace.ingest(self.packet(objects=[self.object(**values)]))

    def select(self):
        snapshot = self.ingest()
        return self.workspace.select_attention(snapshot['objects'][0]['ref'], snapshot['revision'])

    def test_initial_state_is_unknown_not_observed_absent(self):
        value = self.workspace.snapshot()
        self.assertEqual('unknown', value['actors']['presence_state'])
        self.assertEqual([], value['objects'])
        self.assertIsNone(value['attention'])
        self.assertIsNone(value['goal'])
        self.assertFalse(value['privacy']['persistent'])
        self.assertFalse(value['privacy']['automatic_capture'])

    def test_ingestion_creates_anonymous_source_scoped_refs_not_identity(self):
        value = self.workspace.ingest(self.packet(objects=[self.object()], people=[self.person(
            identity={'label': 'SYNTHETIC_NAME', 'profile_id': 'SYNTHETIC_PROFILE'},
            name='SYNTHETIC_NAME', track_id='SYNTHETIC_RAW_TRACK')]))
        dumped = json.dumps(value)
        self.assertNotIn('SYNTHETIC_NAME', dumped)
        self.assertNotIn('SYNTHETIC_PROFILE', dumped)
        self.assertNotIn('SYNTHETIC_RAW_TRACK', dumped)
        self.assertTrue(value['actors']['visible_people'][0]['ref'].startswith('actor_'))
        self.assertTrue(value['objects'][0]['ref'].startswith('object_'))
        self.assertTrue(value['objects'][0]['aggregate'])
        self.assertNotIn('bbox', value['objects'][0])

    def test_voice_touch_and_visible_face_are_never_automatically_bound(self):
        self.workspace.ingest(self.packet(people=[self.person()]))
        for mode in ('voice', 'text', 'touch'):
            value = self.workspace.input('synthetic input', mode)
            actors = value['actors']
            self.assertEqual('unknown', actors['microphone_speaker']['identity'])
            self.assertEqual('unknown', actors['touch_operator']['identity'])
            self.assertFalse(actors['microphone_speaker']['profile_authorized'])
            self.assertFalse(actors['touch_operator']['profile_authorized'])
            self.assertEqual('unproven', actors['cross_modal_identity_link'])

    def test_event_metadata_distinguishes_inputs_and_reply_is_not_delivery(self):
        for mode in ('voice', 'text', 'touch'):
            value = self.workspace.input('RAW_SYNTHETIC_INPUT', mode)
            event = value['events'][-1]
            self.assertEqual('input_received', event['kind'])
            self.assertEqual('input.' + mode, event['source'])
            self.assertEqual(mode, event['input_type'])
        reply = self.workspace.reply('RAW_SYNTHETIC_REPLY')['events'][-1]
        self.assertEqual('agent.output_proposal', reply['source'])
        self.assertNotIn('input_type', reply)
        self.assertNotIn('RAW_SYNTHETIC', json.dumps(self.workspace.snapshot()))

    def test_select_is_a_touch_event_and_rejects_actor_refs(self):
        value = self.workspace.ingest(self.packet(objects=[self.object()], people=[self.person()]))
        with self.assertRaises(ValueError):
            self.workspace.select_attention(value['actors']['visible_people'][0]['ref'], value['revision'])
        selected = self.workspace.select_attention(value['objects'][0]['ref'], value['revision'])
        self.assertEqual('input.touch', selected['events'][-1]['source'])
        self.assertEqual(value['objects'][0], selected['attention'])

    def test_clear_attention_without_focus_is_a_noop(self):
        before = self.workspace.set_goal('conversation', '继续当前对话', 'text')
        self.assertEqual(before, self.workspace.clear_attention())

    def test_clear_attention_preserves_scene_actors_and_transient_dialogue(self):
        scene = self.workspace.ingest(self.packet(objects=[self.object()], people=[self.person()]))
        self.workspace.select_attention(scene['objects'][0]['ref'], scene['revision'])
        self.workspace.input('SYNTHETIC_TRANSIENT_INPUT', 'voice')
        self.workspace.reply('SYNTHETIC_TRANSIENT_REPLY')
        before, dialogue = self.workspace.snapshot(), self.workspace.dialogue()
        value = self.workspace.clear_attention()
        self.assertIsNone(value['attention'])
        self.assertEqual(before['revision'] + 1, value['revision'])
        for key in ('objects', 'actors', 'history_count', 'privacy'):
            self.assertEqual(before[key], value[key])
        self.assertEqual(dialogue, self.workspace.dialogue())
        self.assertEqual('attention_invalidated', value['events'][-1]['kind'])
        self.assertEqual(before['attention']['ref'], value['events'][-1]['ref'])
        self.assertNotIn('SYNTHETIC_TRANSIENT', json.dumps(value))
        self.assertIsNone(self.workspace.packet()['attention'])

    def test_clear_attention_revokes_only_dependent_target_goal(self):
        selected = self.select()
        goal = self.workspace.set_goal('inspect_object', '看看这个', 'touch')['goal']
        value = self.workspace.clear_attention()
        self.assertEqual(goal['id'], value['goal']['id'])
        self.assertEqual('unknown', value['goal']['state'])
        self.assertEqual('target_unavailable', value['goal']['step'])
        self.assertEqual(['object_reference'], value['goal']['missing'])
        self.assertEqual([], value['goal']['object_refs'])
        self.assertGreater(value['goal']['revision'], goal['revision'])
        self.assertEqual(selected['objects'], value['objects'])
        self.assertEqual(['attention_invalidated', 'goal_revoked'],
                         [row['kind'] for row in value['events'][-2:]])

    def test_clear_attention_does_not_change_independent_goal(self):
        for kind, slots in (('conversation', None), ('reminder', {'minutes': 10, 'topic': 'rest'})):
            with self.subTest(kind=kind):
                self.workspace.clear()
                self.select()
                before = self.workspace.set_goal(kind, '可信目标', 'text', slots)
                value = self.workspace.clear_attention()
                self.assertEqual(before['goal'], value['goal'])
                self.assertEqual(before['revision'] + 1, value['revision'])

    def test_cleared_focus_cannot_reappear_on_timestamp_only_refresh(self):
        selected = self.select()
        cleared = self.workspace.clear_attention()
        self.now += 1
        refreshed = self.ingest()
        self.assertIsNone(refreshed['attention'])
        self.assertEqual(cleared['revision'], refreshed['revision'])
        self.assertEqual(selected['objects'][0]['ref'], refreshed['objects'][0]['ref'])

    def test_clear_attention_fences_old_selection_but_allows_fresh_explicit_selection(self):
        selected = self.select()
        ref = selected['attention']['ref']
        cleared = self.workspace.clear_attention()
        with self.assertRaises(ValueError):
            self.workspace.select_attention(ref, selected['revision'])
        reselected = self.workspace.select_attention(ref, cleared['revision'])
        self.assertEqual(ref, reselected['attention']['ref'])
        self.assertEqual(selected['attention']['expires_at'], reselected['attention']['expires_at'])

    def test_clear_attention_does_not_extend_existing_goal_session(self):
        self.select()
        self.workspace.set_goal('inspect_object', '看看这个', 'touch')
        self.now += 5
        value = self.workspace.clear_attention()
        self.assertEqual('unknown', value['goal']['state'])
        self.now += 895
        self.assertIsNone(self.workspace.snapshot()['goal'])

    def test_concurrent_clear_attention_is_idempotent_and_returns_independent_views(self):
        selected = self.select()
        with ThreadPoolExecutor(max_workers=8) as pool:
            views = list(pool.map(lambda _: self.workspace.clear_attention(), range(40)))
        value = self.workspace.snapshot()
        self.assertIsNone(value['attention'])
        self.assertEqual(selected['revision'] + 1, value['revision'])
        self.assertEqual(1, sum(row['kind'] == 'attention_invalidated' for row in value['events']))
        views[0]['objects'][0]['label'] = 'rewritten'
        views[0]['events'].clear()
        self.assertEqual('cup', self.workspace.snapshot()['objects'][0]['label'])
        self.assertTrue(self.workspace.snapshot()['events'])

    def test_timestamp_only_refresh_keeps_semantic_revision_and_ref(self):
        first = self.ingest()
        self.now += 1
        refreshed = self.ingest()
        self.assertEqual(first['revision'], refreshed['revision'])
        self.assertEqual(first['objects'][0]['ref'], refreshed['objects'][0]['ref'])
        self.assertNotEqual(first['objects'][0]['expires_at'], refreshed['objects'][0]['expires_at'])

    def test_refresh_does_not_renew_selected_original_evidence(self):
        selected = self.select()
        expiry = selected['attention']['expires_at']
        self.now += 5
        refreshed = self.ingest()
        self.assertEqual(expiry, refreshed['attention']['expires_at'])
        self.assertGreater(refreshed['objects'][0]['expires_at'], expiry)
        self.now = expiry
        expired = self.workspace.snapshot()
        self.assertIsNone(expired['attention'])
        self.assertTrue(expired['objects'])
        self.assertGreater(expired['revision'], refreshed['revision'])

    def test_new_fresh_evidence_after_original_expiry_does_not_revive_attention(self):
        first = self.select()
        self.now += 11
        refreshed = self.ingest()
        self.assertIsNone(refreshed['attention'])
        self.assertNotEqual(first['objects'][0]['ref'], refreshed['objects'][0]['ref'])

    def test_actual_object_semantic_change_invalidates_attention(self):
        for changes in ({'count': 2}, {'label': 'book'}, {'source': 'camera.front'}):
            with self.subTest(changes=changes):
                self.workspace.clear()
                first = self.select()
                current = self.ingest(**changes)
                self.assertIsNone(current['attention'])
                self.assertGreater(current['revision'], first['revision'])
                self.assertNotEqual(current['objects'][0]['ref'], first['objects'][0]['ref'])

    def test_people_change_removal_and_source_state_change_fence_revision(self):
        first = self.workspace.ingest(self.packet(people=[self.person()]))
        second = self.workspace.ingest(self.packet(people=[self.person(track_id='anonymous-track-2')]))
        self.assertGreater(second['revision'], first['revision'])
        self.assertNotEqual(first['actors']['visible_people'][0]['ref'], second['actors']['visible_people'][0]['ref'])
        unavailable = self.workspace.ingest(self.packet(people_state='unavailable'))
        self.assertGreater(unavailable['revision'], second['revision'])
        unknown = self.workspace.ingest(self.packet(people_state='unknown'))
        self.assertGreater(unknown['revision'], unavailable['revision'])

    def test_disabled_missing_and_malformed_ingest_discard_old_target(self):
        for packet in (None, [], {}, {'observations': []},
                       self.packet(objects=[self.object()], object_state='disabled')):
            with self.subTest(packet=type(packet).__name__):
                self.select()
                result = self.workspace.ingest(packet)
                self.assertEqual([], result['objects'])
                self.assertIsNone(result['attention'])

    def test_future_nan_infinite_bool_and_overlong_ttl_rows_are_rejected(self):
        for values in ({'observed_at': self.now + 1}, {'expires_at': self.now},
                       {'observed_at': float('nan')}, {'expires_at': float('inf')},
                       {'expires_at': 10 ** 10000}, {'observed_at': True},
                       {'expires_at': self.now + 1000}, {'source': 'private/synthetic'},
                       {'count': True}, {'count': 101}, {'label': 'x' * 65}, {'label': '\ud800'}):
            with self.subTest(keys=list(values)):
                result = self.ingest(**values)
                self.assertEqual([], result['objects'])

    def test_conflicting_duplicate_aggregate_counts_fail_closed(self):
        value = self.workspace.ingest(self.packet(objects=[self.object(count=1), self.object(count=2)]))
        self.assertEqual([], value['objects'])

    def test_contradictory_absence_is_unknown_not_an_absence_claim(self):
        value = self.workspace.ingest(self.packet(people=[self.person()], people_state='observed_absent'))
        self.assertEqual('unknown', value['actors']['presence_state'])
        actual_absence = self.workspace.ingest(self.packet(people_state='observed_absent'))
        self.assertEqual('observed_absent', actual_absence['actors']['presence_state'])

    def test_shorter_ref_evidence_expiry_also_invalidates_attention(self):
        selected = self.select()
        self.now += 1
        self.ingest(expires_at=self.now + 2)
        self.assertEqual(selected['attention']['expires_at'], self.workspace.snapshot()['attention']['expires_at'])
        self.now += 2
        expired = self.workspace.snapshot()
        self.assertEqual([], expired['objects'])
        self.assertIsNone(expired['attention'])

    def test_zero_count_is_never_a_selectable_instance(self):
        value = self.ingest(count=0)
        with self.assertRaises(ValueError):
            self.workspace.select_attention(value['objects'][0]['ref'], value['revision'])

    def test_stale_revision_and_expired_reference_cannot_be_selected(self):
        value = self.ingest()
        ref, revision = value['objects'][0]['ref'], value['revision']
        for old_revision in (revision - 1, True, '1', None):
            with self.assertRaises(ValueError):
                self.workspace.select_attention(ref, old_revision)
        self.workspace.input('changed task', 'text')
        with self.assertRaises(ValueError):
            self.workspace.select_attention(ref, revision)
        self.now += 11
        with self.assertRaises(ValueError):
            self.workspace.select_attention(ref, self.workspace.snapshot()['revision'])

    def test_history_is_six_short_entries_and_not_public_diagnostics(self):
        for _ in range(20):
            self.workspace.input('RAW_PRIVATE_SYNTHETIC_INPUT' + 'x' * 500, 'voice')
            self.workspace.reply('RAW_PRIVATE_SYNTHETIC_REPLY' + 'y' * 500)
        dialogue = self.workspace.dialogue()
        self.assertEqual(6, len(dialogue))
        self.assertTrue(all(set(row) == {'role', 'content'} and len(row['content']) <= 240 for row in dialogue))
        self.assertEqual('assistant', dialogue[-1]['role'])
        self.assertNotIn('RAW_PRIVATE', json.dumps(self.workspace.snapshot()))
        self.assertNotIn('RAW_PRIVATE', json.dumps(self.workspace.packet()))
        self.assertEqual(6, self.workspace.snapshot()['history_count'])

    def test_history_expires_without_reads_or_perception_extending_it(self):
        workspace = Workspace(clock=lambda: self.now, history_ttl=10)
        workspace.input('synthetic', 'text')
        before = workspace.snapshot()['revision']
        for _ in range(2):
            self.now += 5
            workspace.snapshot()
        self.assertEqual([], workspace.dialogue())
        self.assertGreater(workspace.snapshot()['revision'], before)

    def test_goal_frame_is_generic_and_never_execution_authority(self):
        result = self.workspace.set_goal('reminder', '稍后提醒休息', 'voice', {'minutes': 10, 'topic': 'rest'})
        goal = result['goal']
        self.assertEqual('reminder', goal['kind'])
        self.assertEqual('proposed', goal['state'])
        self.assertEqual('anonymous_voice_session', goal['initiator_scope'])
        self.assertEqual([], goal['missing'])
        self.assertNotIn('authorized', goal)
        self.assertFalse(result['privacy']['profile_access'])

    def test_missing_slots_and_unknown_object_target_remain_explicit(self):
        first = self.workspace.set_goal('reminder', '提醒', 'touch')
        self.assertEqual(['minutes', 'topic'], first['goal']['missing'])
        second = self.workspace.set_goal('inspect_object', '看看这个', 'voice')
        self.assertEqual(['object_reference'], second['goal']['missing'])
        self.assertEqual([], second['goal']['object_refs'])

    def test_selected_target_goal_revoked_when_attention_original_ttl_expires(self):
        self.select()
        first = self.workspace.set_goal('inspect_object', '看看这个', 'voice')
        self.now += 5
        self.ingest()
        self.now += 5
        result = self.workspace.snapshot()
        self.assertEqual('unknown', result['goal']['state'])
        self.assertEqual(['object_reference'], result['goal']['missing'])
        self.assertEqual([], result['goal']['object_refs'])
        self.assertGreater(result['goal']['revision'], first['goal']['revision'])

    def test_goal_target_revoked_on_semantic_change_not_reassigned(self):
        self.select()
        first = self.workspace.set_goal('inspect_object', '看看这个', 'touch')
        second = self.ingest(count=2)
        self.assertEqual(first['goal']['id'], second['goal']['id'])
        self.assertEqual('unknown', second['goal']['state'])
        self.assertEqual([], second['goal']['object_refs'])

    def test_goal_state_receipt_fencing_and_terminal_no_revival(self):
        first = self.workspace.set_goal('reminder', '休息提醒', 'text', {'minutes': 2, 'topic': 'rest'})['goal']
        updated = self.workspace.set_goal_state('awaiting_confirmation', 'confirm', goal_id=first['id'], revision=first['revision'])['goal']
        with self.assertRaises(ValueError):
            self.workspace.set_goal_state('scheduled', 'scheduled', revision=first['revision'])
        self.workspace.set_goal('conversation', '新目标', 'text')
        with self.assertRaises(ValueError):
            self.workspace.set_goal_state('scheduled', 'scheduled', goal_id=updated['id'])
        self.workspace.set_goal_state('cancelled', 'cancelled')
        with self.assertRaises(ValueError):
            self.workspace.set_goal_state('scheduled', 'scheduled')

    def test_noop_goal_state_update_preserves_revision(self):
        first = self.workspace.set_goal('conversation', '聊一聊', 'text')
        same = self.workspace.set_goal_state('proposed', 'proposed')
        self.assertEqual(first['revision'], same['revision'])

    def test_goal_expiry_not_renewed_by_observations_or_assistant_replies(self):
        self.workspace.set_goal('conversation', '聊一聊', 'text')
        self.now += 899
        self.ingest()
        self.workspace.reply('只是一条候选回复')
        self.now += 1
        self.assertIsNone(self.workspace.snapshot()['goal'])

    def test_explicit_user_activity_extends_goal_session(self):
        self.workspace.set_goal('conversation', '聊一聊', 'text')
        self.now += 899
        self.workspace.input('继续', 'text')
        self.now += 2
        self.assertIsNotNone(self.workspace.snapshot()['goal'])

    def test_invalid_goal_fields_slots_states_and_inputs_are_rejected(self):
        for kind, title, mode, slots in (
            ('shutdown', 'x', 'text', None), ('conversation', '', 'text', None),
            ('conversation', 'x', [], None), ('conversation', 'x', 'text', {'topic': 'rest'}),
            ('reminder', 'x', 'voice', {'minutes': True}), ('reminder', 'x', 'voice', {'minutes': 0}),
            ('reminder', 'x', 'voice', {'minutes': 1441}), ('reminder', 'x', 'voice', {'topic': []}),
            ('reminder', 'x', 'voice', {'topic': 'private'}), ('reminder', 'x', 'voice', {'command': 'unsafe'})):
            with self.assertRaises(ValueError):
                self.workspace.set_goal(kind, title, mode, slots)
        self.workspace.set_goal('conversation', '聊一聊', 'text')
        for state, step, missing in (('execute', 'x', ()), ('proposed', 'x' * 65, ()),
                                     ('proposed', 'x', ('bad/path',)), ('proposed', 'x', ('x',) * 5)):
            with self.assertRaises(ValueError):
                self.workspace.set_goal_state(state, step, missing)
        for mode in ([], None, True, 'camera'):
            with self.assertRaises(ValueError):
                self.workspace.input('x', mode)

    def test_clear_and_clock_rollback_drop_all_old_session_authority(self):
        for reset in ('clear', 'rollback'):
            with self.subTest(reset=reset):
                self.select()
                self.workspace.input('synthetic', 'voice')
                self.workspace.set_goal('inspect_object', '看看这个', 'voice')
                old = self.workspace.snapshot()
                if reset == 'clear':
                    self.workspace.clear()
                else:
                    self.now -= 1
                value = self.workspace.snapshot()
                self.assertGreater(value['revision'], old['revision'])
                self.assertIsNone(value['attention'])
                self.assertIsNone(value['goal'])
                self.assertEqual([], value['objects'])
                self.assertEqual(0, value['history_count'])

    def test_invalid_clock_fails_closed_without_nan_public_state(self):
        self.select()
        self.workspace.input('synthetic', 'text')
        self.now = float('nan')
        with self.assertRaises(ValueError):
            self.workspace.snapshot()
        self.now = 1001.0
        value = self.workspace.snapshot()
        self.assertEqual([], value['objects'])
        self.assertEqual(0, value['history_count'])
        json.dumps(value, allow_nan=False)

    def test_returned_views_and_dialogue_are_independent_copies(self):
        first = self.select()
        first['objects'][0]['label'] = 'rewritten'
        first['attention']['label'] = 'rewritten'
        self.workspace.input('original', 'voice')
        dialogue = self.workspace.dialogue()
        dialogue[0]['content'] = 'rewritten'
        packet = self.workspace.packet()
        packet['actors']['microphone_speaker']['profile_authorized'] = True
        packet['recent_events'].clear()
        current = self.workspace.snapshot()
        self.assertEqual('cup', current['attention']['label'])
        self.assertEqual('original', self.workspace.dialogue()[0]['content'])
        self.assertFalse(current['actors']['microphone_speaker']['profile_authorized'])

    def test_memory_counts_and_public_json_are_bounded(self):
        self.workspace.ingest(self.packet(
            objects=[self.object(label='object%d' % index) for index in range(200)],
            people=[self.person(track_id='track%d' % index) for index in range(200)]))
        for _ in range(100):
            self.workspace.input('bounded', 'text')
        value, packet = self.workspace.snapshot(), self.workspace.packet()
        self.assertEqual(20, len(value['objects']))
        self.assertEqual(8, len(value['actors']['visible_people']))
        self.assertEqual(32, len(value['events']))
        self.assertEqual(8, len(packet['recent_events']))
        self.assertEqual(6, value['history_count'])
        self.assertLess(len(json.dumps(value)), 20000)
        self.assertLess(len(json.dumps(packet)), 6000)

    def test_history_ttl_cannot_disable_expiry(self):
        for value in (0, -1, True, None, float('nan'), float('inf'), 901):
            with self.assertRaises(ValueError):
                Workspace(history_ttl=value)

    def test_concurrent_mutators_remain_bounded_and_fence_old_revision(self):
        before = self.workspace.snapshot()['revision']
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda index: self.workspace.input('message %d' % index, 'text'), range(100)))
        value = self.workspace.snapshot()
        self.assertEqual(before + 100, value['revision'])
        self.assertEqual(6, value['history_count'])
        self.assertEqual(32, len(value['events']))


if __name__ == '__main__':
    unittest.main()
