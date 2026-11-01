"""Bounded, semantic-only interaction state and ambient behaviour arbitration.

No I/O, media, model calls or durable identity memory. Output remains the sole
playback owner. A reservation is permission to attempt an action, not evidence
that it was heard. Only an owned Output receipt establishes a greeting episode.
"""
from collections import Counter, deque
from copy import deepcopy
import math
import threading


TERMINAL = frozenset({'completed', 'cancelled', 'interrupted', 'expired', 'failed'})


def fresh(item, now):
    return bool(item and item.get('state') == 'present' and not item.get('conflict')
                and item.get('observed_at', float('inf')) <= now < item.get('expires_at', 0))


def latest(snapshot, kind, source):
    items = [v for v in snapshot.get('items', [])
             if v.get('kind') == kind and v.get('source') == source]
    return max(items, key=lambda v: v['arrived_at']) if items else None


class InteractionArbiter:
    """Single ambient reservation; user turns/output/quiet always take priority."""
    def __init__(self, clock):
        self.clock = clock
        self.lock = threading.RLock()
        self.boundary = None
        self.pending = None
        self.episode = None
        self.state = {'phase': 'unavailable', 'reason': 'not_observed', 'expires_at': 0}
        self.revision = 0
        self.transitions = deque(maxlen=32)
        self.decisions = deque(maxlen=32)
        self.counts = Counter()

    @staticmethod
    def subject(event):
        # Never retain the identity label or hash. Body continuity is also
        # required for unnamed waves and cannot be borrowed across sources.
        return (event['source'], event['generation'], event['body_epoch'], event['track_id'])

    def _clear(self):
        self.pending = self.episode = None

    def _record(self, kind, action, reason):
        self.counts[action] += 1
        self.decisions.append({'at': self.clock(), 'kind': kind, 'action': action, 'reason': reason})
        return {'allowed': action == 'allow', 'reason': reason}

    def _valid_subject(self, subject, snapshot, live, now):
        source, generation, body_epoch, track = subject
        stamp = live.get('observed_at')
        if (live.get('enabled') is not True or live.get('person_present') is not True
                or live.get('generation') != generation or live.get('body_epoch') != body_epoch
                or type(stamp) not in (int, float) or not math.isfinite(stamp)
                or not 0 <= now - stamp < 2):
            return False
        if track is not None:
            presence = latest(snapshot, 'presence', source)
            return bool(fresh(presence, now) and presence['data'].get('count') == 1
                        and presence['data'].get('track_ids') == [track])
        return True

    def _gate(self, snapshot, now, own_job=None):
        interaction = latest(snapshot, 'interaction', 'conversation')
        output = latest(snapshot, 'output', 'output-arbiter')
        if not fresh(interaction, now):
            return 'unavailable', 'interaction_unknown', now
        data = interaction['data']
        expiry = interaction['expires_at']
        # Missing flags are unknown during a rolling update, never consent.
        if type(data.get('quiet')) is not bool or type(data.get('user_busy')) is not bool:
            return 'unavailable', 'interaction_flags_unknown', expiry
        if data['quiet']:
            return 'quiet', 'quiet_mode', expiry
        phase = data.get('phase')
        if phase in {'listening', 'thinking'}:
            return phase, 'user_turn_active', expiry
        if data.get('turn_id') is not None or data['user_busy']:
            return phase if phase in {'speaking', 'acting'} else 'acting', 'user_turn_active', expiry
        if not fresh(output, now):
            return 'unavailable', 'output_unknown', now
        expiry = min(expiry, output['expires_at'])
        out = output['data']
        own = bool(own_job and out.get('job_id') == own_job and out.get('queued') == 0
                   and out.get('state') in {'synthesizing', 'playing'})
        if out.get('busy') and not own:
            return 'speaking' if out.get('state') == 'playing' else 'acting', 'output_busy', expiry
        if phase != 'idle' and not (own and phase in {'acting', 'speaking'}):
            return 'unavailable' if phase == 'error' else 'acting', 'interaction_not_idle', expiry
        return 'idle', 'available', expiry

    def update(self, snapshot, boundary, live=None):
        with self.lock:
            now = self.clock()
            if boundary != self.boundary:
                self._clear()
                self.boundary = boundary
            if self.pending and self.pending['expires_at'] <= now:
                self.pending = None
            live = live or {}
            if self.episode and (self.episode['until'] <= now
                    or not self._valid_subject(self.episode['subject'], snapshot, live, now)):
                self.episode = None
            if self.pending and self.pending['kind'] == 'wave' and not self._valid_subject(
                    self.pending['subject'], snapshot, live, now):
                self.pending = None
            phase, reason, expiry = self._gate(snapshot, now)
            # New user intent, quiet, broken evidence or session changes must
            # permanently revoke an old ambient ticket, not merely delay it.
            own_job = self.pending['job_id'] if self.pending else None
            _, own_reason, _ = self._gate(snapshot, now, own_job)
            # Output sets current before its first guard RPC reaches us. An
            # unclaimed reservation may therefore observe its own job as busy.
            # Keep it only across this bounded handoff; claim+authorize still
            # require the exact current job, and user/quiet/unknown revoke it.
            handoff = bool(self.pending and own_job is None and own_reason == 'output_busy')
            if own_reason != 'available' and not handoff:
                self.pending = None
            if phase in {'quiet', 'unavailable'}:
                self.episode = None
            if phase == 'idle' and self.pending:
                phase, reason = 'attending', 'ambient_reserved'
                expiry = min(expiry, self.pending['expires_at'])
            elif phase == 'idle' and self.episode:
                phase, reason = 'engaged', 'recent_wave_exchange'
                expiry = min(expiry, self.episode['until'])
            if (phase, reason) != (self.state['phase'], self.state['reason']):
                self.revision += 1
                self.transitions.append({'at': now, 'phase': phase, 'reason': reason})
            self.state = {'phase': phase, 'reason': reason, 'expires_at': expiry}
            return self.read()

    def reserve(self, kind, token, expires_at, snapshot, *, subject=None):
        with self.lock:
            now = self.clock()
            if kind not in {'wave', 'posture'}:
                return self._record('passive', 'suppress', 'context_only')
            _, reason, _ = self._gate(snapshot, now)
            if reason != 'available':
                return self._record(kind, 'suppress', reason)
            if self.pending and self.pending['expires_at'] > now:
                # A wave outranks a posture proposal that has NOT reached
                # Output. Never preempt an admitted/synthesizing/playing job.
                if not (kind == 'wave' and self.pending['kind'] == 'posture'
                        and self.pending['job_id'] is None):
                    return self._record(kind, 'suppress', 'ambient_reserved')
            if kind == 'posture' and self.episode and self.episode['until'] > now:
                return self._record(kind, 'suppress', 'engagement_active')
            continuation = bool(kind == 'wave' and self.episode
                                and self.episode['subject'] == subject and self.episode['until'] > now)
            self.pending = {'kind': kind, 'token': token, 'expires_at': expires_at,
                            'subject': subject, 'job_id': None, 'delivered': False}
            return {**self._record(kind, 'allow', 'continue_engagement' if continuation else 'admitted'),
                    'continuation': continuation}

    def authorize(self, token, job_id, snapshot):
        with self.lock:
            pending = self.pending
            if not pending or pending['token'] != token or pending['expires_at'] <= self.clock():
                return self._record('ambient', 'suppress', 'reservation_revoked')
            if pending['job_id'] not in (None, job_id):
                return self._record(pending['kind'], 'suppress', 'job_mismatch')
            _, reason, _ = self._gate(snapshot, self.clock(), job_id)
            if reason != 'available':
                self.pending = None
                return self._record(pending['kind'], 'suppress', reason)
            pending['job_id'] = job_id
            return {'allowed': True, 'reason': 'arbitration_revalidated'}

    def claim(self, token, job_id):
        """Bind before a status refresh can observe this job as an outsider."""
        with self.lock:
            p = self.pending
            if (not p or p['token'] != token or p['expires_at'] <= self.clock()
                    or not isinstance(job_id, str) or not 1 <= len(job_id) <= 128
                    or p['job_id'] not in (None, job_id)):
                return False
            p['job_id'] = job_id
            return True

    def receipt(self, job_id, state):
        with self.lock:
            p = self.pending
            if not p or p['job_id'] != job_id or job_id is None:
                return
            if state in {'playing', 'completed'} and not p['delivered']:
                p['delivered'] = True
                self.counts['delivered'] += 1
                if p['kind'] == 'wave':
                    self.episode = {'subject': p['subject'], 'until': self.clock() + 60}
            if state in TERMINAL:
                self.counts['terminal_' + state] += 1
                self.pending = None

    def revoke(self, token):
        with self.lock:
            if self.pending and self.pending['token'] == token:
                self.pending = None

    def read(self):
        with self.lock:
            state = dict(self.state)
            if self.clock() >= state['expires_at']:
                state.update(phase='unavailable', reason='interaction_state_stale')
            return {'schema_version': 1, **state, 'revision': self.revision,
                    'microphone_started': False, 'identity_retained': False,
                    'pending_kind': self.pending['kind'] if self.pending else None,
                    'counts': dict(self.counts), 'transitions': deepcopy(list(self.transitions)),
                    'decisions': deepcopy(list(self.decisions))}

    def close(self):
        with self.lock:
            self._clear()
            self.state = {'phase': 'unavailable', 'reason': 'closed', 'expires_at': 0}
