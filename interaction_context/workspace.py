"""Pure, ephemeral multimodal workspace: references and evidence, not identity.

Only already-admitted semantic packets enter this reducer. It does not capture,
run models, execute tools, read profiles, or persist data. Aggregate detections
are never promoted to tracked object instances or spatial/bounding-box claims.
All user-supplied dialogue and goal text remains untrusted model input.
"""

from collections import deque
from copy import deepcopy
from itertools import count
import math
import re
import threading
import time


_NAMESPACES = count(1)
_INPUTS = frozenset({'text', 'voice', 'touch'})
_SOURCE_STATES = frozenset({'available', 'present', 'observed_absent', 'unknown', 'stale',
                            'unavailable', 'disabled', 'uncertain', 'unconfirmed'})
_GOAL_KINDS = frozenset({'reminder', 'inspect_object', 'conversation'})
_GOAL_STATES = frozenset({'proposed', 'awaiting_confirmation', 'submitting', 'scheduled',
                         'cancelled', 'unknown', 'failed', 'completed'})
_TERMINAL = frozenset({'cancelled', 'failed', 'completed'})
_SOURCE = re.compile(r'^[a-z][a-z0-9_.:-]{0,79}$')
_MISSING = re.compile(r'^[a-z][a-z0-9_]{0,47}$')


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _text(value, limit, *, truncate=False):
    if (type(value) is not str or not value.strip()
            or any(ord(char) < 32 and char not in '\n\t' for char in value)
            or len(value) > (1200 if truncate else limit)):
        raise ValueError('invalid bounded text')
    return value.strip()[:limit].encode('utf-8', 'replace').decode('utf-8')


class Workspace:
    """One anonymous interaction session, six short messages, one goal.

    ``revision`` fences semantic authority. Timestamp-only refresh is a no-op;
    actual perceptual/source-state changes, input, selection, goal changes and
    expiry revoke old revisions. Reads expire state but never refresh its TTL.
    """

    def __init__(self, clock=time.time, *, history_ttl=900):
        if not _finite(history_ttl) or not 0 < history_ttl <= 900:
            raise ValueError('history_ttl must be in (0, 900]')
        self._clock = clock
        self._history_ttl = float(history_ttl)
        self._lock = threading.RLock()
        self._namespace = next(_NAMESPACES)
        self._serial = self._event_seq = self._revision = 0
        self._last_clock = None
        self._reset()

    def _reset(self):
        self._people = {}
        self._objects = {}
        self._sources = {'people': 'unknown', 'objects': 'unknown'}
        self._attention = None
        self._goal = None
        self._goal_expires = None
        self._goal_attention_expires = None
        self._history = deque(maxlen=6)
        self._events = deque(maxlen=32)

    def _ref(self, kind):
        self._serial += 1
        return '%s_%d_%d' % (kind, self._namespace, self._serial)

    def _change(self, kind, now, ref=None, *, source=None, input_type=None):
        self._revision += 1
        self._event_seq += 1
        event = {'seq': self._event_seq, 'at': now, 'kind': kind, 'revision': self._revision}
        if ref is not None:
            event['ref'] = ref
        if source is not None:
            event['source'] = source
        if input_type is not None:
            event['input_type'] = input_type
        self._events.append(event)

    def _now(self):
        now = self._clock()
        if not _finite(now):
            previous = self._last_clock if self._last_clock is not None else 0.0
            self._reset()
            self._change('clock_reset', previous)
            raise ValueError('clock must be finite')
        if self._last_clock is not None and now < self._last_clock:
            self._reset()
            self._change('clock_reset', now)
        self._last_clock = now
        return now

    def _revoke_target(self, now):
        if (self._goal is not None and self._goal['object_refs']
                and self._goal['state'] not in _TERMINAL):
            self._goal.update(state='unknown', step='target_unavailable',
                              missing=['object_reference'], object_refs=[])
            self._goal_attention_expires = None
            self._change('goal_revoked', now, self._goal['id'])
            self._goal['revision'] = self._revision

    def _expire(self, now):
        old_people, old_objects = len(self._people), len(self._objects)
        self._people = {key: value for key, value in self._people.items() if now < value['expires_at']}
        self._objects = {key: value for key, value in self._objects.items() if now < value['expires_at']}
        if (len(self._people), len(self._objects)) != (old_people, old_objects):
            if old_people and not self._people:
                self._sources['people'] = 'stale'
            if old_objects and not self._objects:
                self._sources['objects'] = 'stale'
            self._change('perception_changed', now)
        current_refs = {row['ref'] for row in self._objects.values()}
        if (self._attention is not None and (now >= self._attention['expires_at']
                                             or self._attention['ref'] not in current_refs)):
            ref = self._attention['ref']
            self._attention = None
            self._change('attention_expired', now, ref)
            self._revoke_target(now)
        if self._goal_attention_expires is not None and now >= self._goal_attention_expires:
            self._revoke_target(now)
        remaining = [row for row in self._history if now < row['expires_at']]
        if len(remaining) != len(self._history):
            self._history = deque(remaining, maxlen=6)
            self._change('history_expired', now)
        # Goal lifetime depends on explicit session activity, never perception
        # polling or assistant replies. Scheduled is not proof of completion.
        if self._goal is not None and self._goal_expires is not None and now >= self._goal_expires:
            ref = self._goal['id']
            self._goal = None
            self._goal_expires = self._goal_attention_expires = None
            self._change('goal_revoked', now, ref)
        while self._events and now - self._events[0]['at'] >= 900:
            self._events.popleft()

    @staticmethod
    def _fresh(row, now):
        return (type(row) is dict and _finite(row.get('observed_at'))
                and _finite(row.get('expires_at'))
                and row['observed_at'] <= now < row['expires_at']
                and 0 < row['expires_at'] - row['observed_at'] <= 900
                and type(row.get('source')) is str and bool(_SOURCE.fullmatch(row['source'])))

    @staticmethod
    def _state(value):
        value = value.get('state') if type(value) is dict else value
        return value if type(value) is str and value in _SOURCE_STATES else 'unknown'

    def _actors(self):
        unknown = {'identity': 'unknown', 'profile_authorized': False}
        return {'visible_people': deepcopy(list(self._people.values())),
                'presence_state': self._sources['people'],
                'microphone_speaker': dict(unknown), 'touch_operator': dict(unknown),
                'cross_modal_identity_link': 'unproven'}

    def _snapshot(self):
        return {'revision': self._revision, 'actors': self._actors(),
                'objects': deepcopy(list(self._objects.values())),
                'attention': deepcopy(self._attention), 'goal': deepcopy(self._goal),
                'events': deepcopy(list(self._events)), 'history_count': len(self._history),
                'privacy': {'persistent': False, 'profile_access': False,
                            'identity_binding': False, 'automatic_capture': False,
                            'transient_history': True, 'history_ttl_seconds': self._history_ttl,
                            'objects_state': self._sources['objects']}}

    def ingest(self, reasoning):
        """Replace the anonymous view from an already-admitted semantic packet.

        Unknown, disabled, malformed or missing sources never preserve an old
        selected target. Exact duplicate refreshes do not bump the revision.
        """
        with self._lock:
            now = self._now()
            self._expire(now)
            packet = reasoning if type(reasoning) is dict else {}
            observations = packet.get('observations') if type(packet.get('observations')) is dict else {}
            states = packet.get('source_states') if type(packet.get('source_states')) is dict else {}
            new_states = {kind: self._state(states.get(kind)) for kind in ('people', 'objects')}
            people, objects = {}, {}
            raw_people, raw_objects = observations.get('people'), observations.get('objects')
            if new_states['people'] in {'available', 'present'} and type(raw_people) is list:
                for row in raw_people[:8]:
                    track = row.get('track_id') if type(row) is dict else None
                    if not self._fresh(row, now) or type(track) is not str or not 0 < len(track) <= 96:
                        continue
                    key = (row['source'], track)
                    if key in people:
                        continue
                    old = self._people.get(key)
                    people[key] = {'ref': old['ref'] if old else self._ref('actor'), 'source': row['source'],
                                   'observed_at': row['observed_at'], 'expires_at': row['expires_at']}
            if new_states['objects'] in {'available', 'present'} and type(raw_objects) is list:
                conflicts = set()
                for row in raw_objects[:20]:
                    label = row.get('label') if type(row) is dict else None
                    amount = row.get('count') if type(row) is dict else None
                    if (not self._fresh(row, now) or type(label) is not str or not 0 < len(label) <= 64
                            or not label.strip() or any(ord(char) < 32 for char in label)
                            or type(amount) is not int or not 0 <= amount <= 100):
                        continue
                    try:
                        label.encode('utf-8')
                    except UnicodeEncodeError:
                        continue
                    key = (row['source'], label)
                    if key in objects:
                        if objects[key]['count'] != amount:
                            conflicts.add(key)
                        continue
                    old = self._objects.get(key)
                    objects[key] = {'ref': old['ref'] if old and old['count'] == amount else self._ref('object'),
                                    'source': row['source'], 'label': label, 'count': amount, 'aggregate': True,
                                    'observed_at': row['observed_at'], 'expires_at': row['expires_at']}
                for key in conflicts:
                    objects.pop(key, None)
            for kind, raw, admitted in (('people', raw_people, people), ('objects', raw_objects, objects)):
                if new_states[kind] in {'available', 'present'} and (type(raw) is not list or not admitted):
                    new_states[kind] = 'unknown'
                elif new_states[kind] == 'observed_absent' and type(raw) is list and raw:
                    # Contradictory supplied facts do not establish absence.
                    new_states[kind] = 'unknown'
            signature = lambda people, objects, states: (
                sorted((key, value['ref']) for key, value in people.items()),
                sorted((key, value['ref'], value['count']) for key, value in objects.items()), states)
            changed = signature(self._people, self._objects, self._sources) != signature(people, objects, new_states)
            self._people, self._objects, self._sources = people, objects, new_states
            if changed:
                self._change('perception_changed', now)
            if self._attention is not None:
                current_refs = {row['ref'] for row in objects.values()}
                if self._attention['ref'] not in current_refs:
                    ref = self._attention['ref']
                    self._attention = None
                    self._change('attention_invalidated', now, ref)
                    self._revoke_target(now)
            if self._goal is not None and self._goal['object_refs']:
                if not set(self._goal['object_refs']) <= {row['ref'] for row in objects.values()}:
                    self._revoke_target(now)
            return self._snapshot()

    def select_attention(self, ref, revision):
        with self._lock:
            now = self._now()
            self._expire(now)
            if type(ref) is not str or type(revision) is not int or revision != self._revision:
                raise ValueError('stale object selection')
            selected = next((row for row in self._objects.values() if row['ref'] == ref and row['count'] > 0), None)
            if selected is None:
                raise ValueError('unknown current aggregate object reference')
            self._attention = deepcopy(selected)
            self._change('attention_selected', now, ref, source='input.touch', input_type='touch')
            if self._goal is not None:
                self._goal_expires = now + 900
            return self._snapshot()

    def clear_attention(self):
        """Drop shared object focus without discarding its source or dialogue.

        An explicit task selection may replace the shared focus while the
        anonymous scene remains available. Only goals depending on the old
        selected evidence are revoked; clearing never renews evidence or goals.
        """
        with self._lock:
            now = self._now()
            self._expire(now)
            if self._attention is not None:
                ref = self._attention['ref']
                self._attention = None
                self._change('attention_invalidated', now, ref)
                self._revoke_target(now)
            return self._snapshot()

    def _message(self, message, role, now, input_type=None):
        content = _text(message, 240, truncate=True)
        self._history.append({'role': role, 'content': content, 'expires_at': now + self._history_ttl})
        self._change('input_received' if role == 'user' else 'reply_recorded', now,
                     source='input.' + input_type if role == 'user' else 'agent.output_proposal',
                     input_type=input_type)

    def input(self, message, input_type):
        if type(input_type) is not str or input_type not in _INPUTS:
            raise ValueError('invalid input type')
        _text(message, 240, truncate=True)
        with self._lock:
            now = self._now()
            self._expire(now)
            self._message(message, 'user', now, input_type)
            if self._goal is not None:
                self._goal_expires = now + 900
            return self._snapshot()

    def reply(self, text):
        _text(text, 240, truncate=True)
        with self._lock:
            now = self._now()
            self._expire(now)
            self._message(text, 'assistant', now)
            return self._snapshot()

    def dialogue(self):
        with self._lock:
            self._expire(self._now())
            return [{'role': row['role'], 'content': row['content']} for row in self._history]

    @staticmethod
    def _slots(kind, slots):
        slots = {} if slots is None else slots
        if type(slots) is not dict or set(slots) - {'minutes', 'topic'}:
            raise ValueError('invalid goal slots')
        if kind != 'reminder' and slots:
            raise ValueError('goal has no typed slots')
        if ('minutes' in slots and (type(slots['minutes']) is not int or not 1 <= slots['minutes'] <= 1440)
                or 'topic' in slots and (type(slots['topic']) is not str
                                         or slots['topic'] not in {'rest', 'water', 'move'})):
            raise ValueError('invalid reminder slots')
        return dict(slots)

    def set_goal(self, kind, title, input_type, slots=None):
        if (type(kind) is not str or kind not in _GOAL_KINDS
                or type(input_type) is not str or input_type not in _INPUTS):
            raise ValueError('invalid goal scope')
        title, slots = _text(title, 120), self._slots(kind, slots)
        with self._lock:
            now = self._now()
            self._expire(now)
            object_refs = [self._attention['ref']] if kind == 'inspect_object' and self._attention else []
            missing = ([key for key in ('minutes', 'topic') if key not in slots] if kind == 'reminder'
                       else ['object_reference'] if kind == 'inspect_object' and not object_refs else [])
            ref = self._ref('goal')
            self._change('goal_created', now, ref)
            self._goal = {'id': ref, 'revision': self._revision, 'kind': kind, 'title': title,
                          'state': 'proposed', 'step': 'clarify' if missing else 'proposed',
                          'missing': missing, 'initiator_scope': 'anonymous_%s_session' % input_type,
                          'object_refs': object_refs, 'slots': slots}
            self._goal_expires = now + 900
            self._goal_attention_expires = self._attention['expires_at'] if object_refs else None
            return self._snapshot()

    def set_goal_state(self, state, step, missing=(), *, goal_id=None, revision=None):
        if type(state) is not str or state not in _GOAL_STATES:
            raise ValueError('invalid goal state')
        step = _text(step, 64)
        if (type(missing) not in (list, tuple) or len(missing) > 4
                or any(type(value) is not str or not _MISSING.fullmatch(value) for value in missing)):
            raise ValueError('invalid missing fields')
        with self._lock:
            now = self._now()
            self._expire(now)
            if (self._goal is None or goal_id is not None and goal_id != self._goal['id']
                    or revision is not None and (type(revision) is not int or revision != self._goal['revision'])):
                raise ValueError('stale goal update')
            if self._goal['state'] in _TERMINAL and state != self._goal['state']:
                raise ValueError('terminal goal cannot be revived')
            if (state, step, list(missing)) != (self._goal['state'], self._goal['step'], self._goal['missing']):
                self._change('goal_changed', now, self._goal['id'])
                self._goal.update(state=state, step=step, missing=list(missing), revision=self._revision)
            return self._snapshot()

    def clear(self):
        with self._lock:
            now = self._now()
            self._reset()
            self._change('session_cleared', now)
            return self._snapshot()

    def snapshot(self):
        with self._lock:
            self._expire(self._now())
            return self._snapshot()

    def packet(self):
        with self._lock:
            self._expire(self._now())
            return {'goal': deepcopy(self._goal), 'attention': deepcopy(self._attention),
                    'actors': self._actors(), 'recent_events': deepcopy(list(self._events)[-8:])}
