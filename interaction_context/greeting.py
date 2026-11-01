"""Short-lived gesture evidence and deterministic greeting admission; no media/history."""
import hashlib
import math
import re
import secrets
from collections import OrderedDict

from .core import Access


def valid_event(event, now):
    fields = {'kind', 'event_id', 'source', 'observed_at', 'confidence',
              'track_id', 'identity_key', 'generation', 'body_epoch'}
    if not isinstance(event, dict) or set(event) != fields:
        raise ValueError('invalid greeting event fields')
    if event['kind'] not in {'wave', 'arrival'} or event['source'] not in {'camera.front', 'camera.environment'}:
        raise ValueError('invalid greeting source')
    if event['kind'] == 'wave' and event['source'] != 'camera.front':
        raise ValueError('wave source must be front')
    if not isinstance(event['event_id'], str) or not re.fullmatch(r'[a-f0-9]{32}', event['event_id']):
        raise ValueError('invalid greeting id')
    for key in ('observed_at', 'confidence'):
        if type(event[key]) not in (int, float) or not math.isfinite(event[key]):
            raise ValueError('invalid greeting number')
    if not 0 <= now - event['observed_at'] < 3 or not .75 <= event['confidence'] <= 1:
        raise ValueError('stale or uncertain greeting')
    if type(event['generation']) is not int or event['generation'] < 0:
        raise ValueError('invalid greeting generation')
    if event['kind'] == 'wave' and (type(event['body_epoch']) is not int or event['body_epoch'] < 0):
        raise ValueError('invalid body continuity')
    if event['track_id'] is not None and (not isinstance(event['track_id'], str)
            or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', event['track_id'])):
        raise ValueError('invalid subject')
    if event['identity_key'] is not None and (not event['track_id']
            or not isinstance(event['identity_key'], str)
            or not re.fullmatch(r'[a-f0-9]{64}', event['identity_key'])):
        raise ValueError('invalid identity binding')


def decide(snapshot, event, live, now, *, own_job=None):
    """Evidence fusion, not a VLM instruction or a camera's direct speech trigger."""
    # Passive context never grants speech, including legacy arrival tickets.
    if event.get('kind') != 'wave':
        return 'context_only', None
    def latest(kind, source):
        values = [v for v in snapshot['items'] if v['kind'] == kind and v['source'] == source]
        return max(values, key=lambda v: v['arrived_at']) if values else None
    def fresh(v):
        return v is not None and v['state'] == 'present' and v['observed_at'] <= now < v['expires_at']
    if live.get('generation') != event['generation']:
        return 'generation_changed', None
    if live.get('enabled' if event['kind'] == 'wave' else 'arrival_enabled') is not True:
        return 'visual_interaction_off', None
    if live.get('quiet') is not False:
        return 'quiet_or_unavailable', None
    if live.get('user_busy') is not False or live.get('turn_active') is not False:
        return 'conversation_busy', None
    if event['kind'] == 'wave' and (live.get('person_present') is not True
            or type(live.get('observed_at')) not in (int, float)
            or not 0 <= now-live['observed_at'] < 2
            or live.get('body_epoch') != event['body_epoch']):
        return 'body_missing_or_changed', None
    interaction = latest('interaction', 'conversation')
    output = latest('output', 'output-arbiter')
    resources = latest('resources', 'control-plane')
    for value, reason in ((interaction, 'interaction_unknown'), (output, 'output_unknown'),
                          (resources, 'resources_unknown')):
        if not fresh(value):
            return reason, None
    own_output = bool(own_job and output['data'].get('job_id') == own_job
                      and output['data'].get('queued') == 0
                      and output['data'].get('state') in {'synthesizing', 'playing'})
    if interaction['data'].get('phase') != 'idle' and not (
            own_output and live.get('output_job') == own_job
            and live.get('user_busy') is False and live.get('turn_active') is False):
        return 'conversation_busy', None
    if output['data'].get('busy') and not own_output:
        return 'robot_speaking', None
    if not resources['data'].get('audio_available') and not own_output:
        return 'audio_busy', None
    if resources['data'].get('cpu_available', 0) <= 0:
        return 'compute_busy', None
    name = None
    presence = latest('presence', event['source'])
    if event['kind'] == 'arrival' and (not fresh(presence) or presence['data'].get('count', 0) < 1):
        return 'arrival_unconfirmed', None
    if event['track_id'] and (not fresh(presence) or presence['data'].get('count') != 1
            or presence['data'].get('track_ids') != [event['track_id']]):
        # A bound person's departure/change invalidates the whole pending event.
        return 'subject_changed', None
    if event['identity_key'] and event['source'] == 'camera.front':
        identities = [v for v in snapshot['items'] if v['kind'] == 'identity'
                      and v['source'] == 'camera.front' and v['subject'] == event['track_id'] and fresh(v)
                      and v['data'].get('consent') == 'enrolled_local']
        if len(identities) == 1:
            label = identities[0]['data'].get('label', '')
            if (isinstance(label, str) and 1 <= len(label) <= 32
                    and all(c.isalnum() or c in ' -' for c in label)
                    and hashlib.sha256(label.encode()).hexdigest() == event['identity_key']):
                name = label
    return 'greet_named' if name else 'greet_generic', name


class GreetingFusion:
    def __init__(self, host):
        self.host = host
        self.seen = OrderedDict()
        self.tickets = OrderedDict()
        self.cooldown_until = 0
        self.count = 0
        self.last_reason = 'not_requested'

    def prune(self):
        now = self.host.clock()
        self.seen = OrderedDict((k, v) for k, v in self.seen.items() if v > now)
        self.tickets = OrderedDict((k, v) for k, v in self.tickets.items() if v['expires_at'] > now)

    def _refresh(self):
        h = self.host
        semantic = h.semantic_provider() if h.semantic_provider else h.last_semantic
        h._ingest_semantic(semantic)
        h._poll_interaction()
        h._poll_output()
        h._poll_resources()
        return h.core.snapshot(Access(session_id=h.active_session, identity=True))

    def _live(self):
        return self.host._read_json('http://127.0.0.1:8765/api/wave/status')

    def propose(self, event):
        h = self.host
        with h.lock:
            now = h.clock()
            valid_event(event, now)
            self.prune()
            if event['event_id'] in self.seen:
                return {'action': 'suppress', 'reason': 'duplicate_event'}
            self.seen[event['event_id']] = now + 30
            while len(self.seen) > 128:
                self.seen.popitem(last=False)
            h.core.observe(provider='pose-wave' if event['kind'] == 'wave' else 'vision-arrival',
                           kind='gesture', source=event['source'], subject=event['track_id'],
                           observed_at=event['observed_at'], ttl_seconds=3, quality=event['confidence'],
                           data={'label': event['kind'], 'event_id': event['event_id'],
                                 'track_id': event['track_id']}, origin='semantic')
            self.count += 1
            if event['kind'] != 'wave':
                self.last_reason = 'context_only'
                return {'action': 'suppress', 'reason': 'context_only'}
            reason, name = 'cooldown', None
            if h.closed:
                reason = 'fusion_closed'
            elif now >= self.cooldown_until:
                try:
                    snapshot = self._refresh()
                    live = self._live()
                    h.interaction.update(snapshot, h._interaction_boundary(), live)
                    reason, name = decide(snapshot, event, live, h.clock())
                    if h.clock() - event['observed_at'] >= 3:
                        reason = 'event_expired'
                except Exception:
                    reason = 'fusion_unavailable'
            self.last_reason = reason
            if reason not in {'greet_named', 'greet_generic'}:
                return {'action': 'suppress', 'reason': reason}
            token = secrets.token_urlsafe(24)
            admission = h.interaction.reserve('wave', token, h.clock() + 12, snapshot,
                                             subject=h.interaction.subject(event))
            if not admission['allowed']:
                self.last_reason = admission['reason']
                return {'action': 'suppress', 'reason': admission['reason']}
            self.cooldown_until = h.clock() + 20
            self.tickets[token] = {'event': dict(event), 'expires_at': h.clock() + 12,
                                   'session_id': h.active_session, 'name': name, 'job_id': None}
            while len(self.tickets) > 8:
                self.tickets.popitem(last=False)
            text = f'你好{name}，我在呢！有什么事吗？' if name else '你好呀，我在呢！有什么事吗？'
            if admission['continuation']:
                text = '我在呢，你说。'
            return {'action': 'greet', 'reason': reason, 'text': text,
                    'response_style': 'continuation' if admission['continuation'] else 'greeting',
                    'guard': {'kind': 'greeting', 'token': token}, 'expires_in_seconds': 12}

    def authorize(self, guard, job_id):
        h = self.host
        with h.lock:
            if (not isinstance(guard, dict) or set(guard) != {'kind', 'token'}
                    or guard['kind'] != 'greeting' or not isinstance(guard['token'], str)
                    or not re.fullmatch(r'[A-Za-z0-9_-]{20,64}', guard['token'])):
                return {'allowed': False, 'reason': 'invalid_greeting_guard'}
            ticket = self.tickets.get(guard.get('token')) if isinstance(guard, dict) else None
            if (h.closed or not ticket or ticket['expires_at'] <= h.clock()
                    or not isinstance(job_id, str) or not 1 <= len(job_id) <= 128
                    or ticket['job_id'] not in (None, job_id)):
                return {'allowed': False, 'reason': 'expired_or_invalid_greeting'}
            try:
                if not h.interaction.claim(guard['token'], job_id):
                    return {'allowed': False, 'reason': 'reservation_revoked'}
                snapshot = self._refresh()
                live = self._live()
                h.interaction.update(snapshot, h._interaction_boundary(), live)
                admission = h.interaction.authorize(guard['token'], job_id, snapshot)
                reason, name = decide(snapshot, ticket['event'], live, h.clock(), own_job=job_id)
            except Exception:
                h.interaction.revoke(guard['token'])
                return {'allowed': False, 'reason': 'fusion_unavailable'}
            allowed = (admission['allowed'] and ticket['expires_at'] > h.clock() and h.active_session == ticket['session_id']
                       and reason in {'greet_named', 'greet_generic'} and name == ticket['name'])
            if allowed:
                ticket['job_id'] = job_id
            else:
                h.interaction.revoke(guard['token'])
            return {'allowed': allowed, 'reason': 'revalidated' if allowed else 'greeting_context_changed'}
