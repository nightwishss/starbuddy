"""Temporal raised-wrist detector. Only a bounded in-memory landmark window is kept."""
import hashlib
import math
import secrets
from collections import deque


def iou(a, b):
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)) or len(a) != 4 or len(b) != 4:
        return 0
    if not all(type(v) in (int, float) and math.isfinite(v) for v in (*a, *b)):
        return 0
    x, y = max(a[0], b[0]), max(a[1], b[1])
    area = max(0, min(a[0]+a[2], b[0]+b[2])-x) * max(0, min(a[1]+a[3], b[1]+b[3])-y)
    union = a[2]*a[3]+b[2]*b[3]-area
    return area/union if union > 0 else 0


def bind(landmarks, face, identity, now):
    """Require a visible pose head inside one fresh face; identity must share its box."""
    sources = [v for v in face.get('sources', []) if v.get('source') == 'camera.front']
    if len(sources) != 1:
        return None, None
    source = sources[0]
    if not source.get('ready') or not 0 <= now-source.get('last_frame_at', 0) < .8:
        return None, None
    tracks = source.get('tracks', [])
    if len(tracks) != 1:
        return None, None
    track = tracks[0]
    if track.get('state') != 'confirmed' or track.get('last_seen_seconds_ago', 99) > .4:
        return None, None
    box = track.get('bbox')
    if not isinstance(box, list) or len(box) != 4:
        return None, None
    x, y, w, h = box
    for index in (0, 2, 5):
        p = landmarks[index]
        if p.visibility < .8 or not (x <= p.x <= x+w and y <= p.y <= y+h):
            return None, None
    key = None
    obs = identity.get('observation') or {}
    age = identity.get('last_frame_seconds_ago')
    label = obs.get('label')
    if (identity.get('ready') and identity.get('source') == 'camera.front'
            and type(age) in (int, float) and 0 <= age < 1.2
            and obs.get('status') == 'known' and isinstance(label, str)
            and iou(box, obs.get('bbox')) >= .45):
        key = hashlib.sha256(label.encode()).hexdigest()
    return track['track_id'], key


class WaveDetector:
    def __init__(self):
        self.points = {15: deque(maxlen=24), 16: deque(maxlen=24)}
        self.subject = None
        self.last_center = None
        self.last_time = None
        self.next_at = 0
        self.epoch = 0
        self.diagnostics = {'last_reason': 'initial', 'longest_run': 0}

    def note(self, reason):
        self.diagnostics['last_reason'] = reason
        self.diagnostics[reason] = self.diagnostics.get(reason, 0)+1

    def reset(self, reason='no_frame'):
        self.note(reason)
        self.epoch += 1
        for q in self.points.values():
            q.clear()
        self.last_center = self.last_time = self.subject = None

    def body(self, poses, face, identity, now, *, association_at=None):
        if len(poses) != 1 or len(poses[0]) < 33:
            self.reset('no_single_pose')
            return None
        p = poses[0]
        if any(v.visibility < .8 or not math.isfinite(v.x+v.y) for v in (p[0], p[11], p[12])):
            self.reset('upper_body_uncertain')
            return None
        width = abs(p[11].x-p[12].x)
        if not .08 < width < .8:
            self.reset('shoulder_scale')
            return None
        center = ((p[11].x+p[12].x)/2, (p[11].y+p[12].y)/2)
        # Status is fetched after inference. Its timestamp can be newer than
        # capture time; compare freshness to receipt time, not the older frame.
        track, identity_key = bind(p, face, identity, now if association_at is None else association_at)
        if self.last_time is not None:
            if not 0 < now-self.last_time < .6:
                self.reset('frame_gap')
            elif math.dist(center, self.last_center) > .15:
                self.reset('body_jump')
            elif track is not None and self.subject is not None and track != self.subject:
                self.reset('person_change')
        # A temporarily unavailable face does not mean the visible body changed.
        # Keep the last bound track solely for detecting a genuine later change.
        self.last_time, self.last_center = now, center
        if track is not None:
            self.subject = track
        return p, track, identity_key, width, center

    def feed(self, poses, face, identity, now, generation, *, association_at=None):
        body = self.body(poses, face, identity, now, association_at=association_at)
        if body is None:
            return None
        p, track, identity_key, width, center = body
        if now < self.next_at:
            self.note('cooldown')
            return None
        self.note('tracking')
        for wrist, elbow, shoulder in ((15, 13, 11), (16, 14, 12)):
            q = self.points[wrist]
            if (not math.isfinite(p[wrist].x+p[wrist].y)
                    or not 0 <= p[wrist].x <= 1 or not 0 <= p[wrist].y <= 1):
                q.clear()
                self.note('hand_outside')
                continue
            if p[wrist].y > p[shoulder].y+.02:
                q.clear()
                self.note('hand_lowered')
                continue
            if p[wrist].visibility < .75 or p[elbow].visibility < .5:
                # Brief self-occlusion must not erase a visible body's whole
                # wave. Do not invent/interpolate any samples; a long gap resets.
                if q and now-q[-1][0] > .4:
                    q.clear()
                self.note('hand_uncertain')
                continue
            if q and now-q[-1][0] > .4:
                q.clear()
            q.append((now, (p[wrist].x-center[0])/width))
            self.diagnostics['longest_run'] = max(self.diagnostics['longest_run'], len(q))
            while q and now-q[0][0] > 3:
                q.popleft()
            if len(q) < 6 or now-q[0][0] < .65:
                continue
            # Count substantial alternating excursions, not frame-to-frame jitter.
            pivot = extreme = q[0][1]
            direction = reversals = 0
            for _, x in list(q)[1:]:
                if direction == 0 and abs(x-pivot) >= .30:
                    direction, extreme = (1 if x > pivot else -1), x
                elif direction and (x-extreme)*direction > 0:
                    extreme = x
                elif direction and (extreme-x)*direction >= .30:
                    reversals += 1
                    direction, extreme = -direction, x
            if reversals >= 2 and max(x for _, x in q)-min(x for _, x in q) >= .5:
                self.next_at = now+4
                self.note('wave')
                for points in self.points.values():
                    points.clear()
                return {'kind': 'wave', 'event_id': secrets.token_hex(16), 'source': 'camera.front',
                        'observed_at': now, 'confidence': min(p[wrist].visibility,
                                                             p[11].visibility, p[12].visibility),
                        'track_id': track, 'identity_key': identity_key, 'generation': generation,
                        'body_epoch': self.epoch}
        return None


class HandWaveDetector(WaveDetector):
    """Open-palm rotation OR lateral travel; pose is only a person/binding guard."""
    MIN_SAMPLES = 4
    MAX_HAND_GAP = .7

    def __init__(self):
        super().__init__()
        self.hand_points = deque(maxlen=32)
        self.diagnostics.update(current_reason='initial', current_blockers=[], active_samples=0)

    def note(self, reason):
        super().note(reason)
        self.diagnostics['current_reason'] = reason
        self.diagnostics['active_samples'] = len(self.hand_points)

    def reset(self, reason='no_frame'):
        super().reset(reason)
        self.hand_points.clear()
        self.diagnostics.update(current_blockers=[], active_samples=0)

    @staticmethod
    def reversals(values, excursion, span):
        if max(values)-min(values) < span:
            return False
        pivot = extreme = values[0]
        direction = turns = 0
        for value in values[1:]:
            if direction == 0 and abs(value-pivot) >= excursion:
                direction, extreme = (1 if value > pivot else -1), value
            elif direction and (value-extreme)*direction > 0:
                extreme = value
            elif direction and (extreme-value)*direction >= excursion:
                turns += 1
                direction, extreme = -direction, value
        return turns >= 2

    def feed_hands(self, poses, hands, face, identity, now, generation, *, association_at=None):
        self.diagnostics.update(current_reason='initial', current_blockers=[],
                                active_samples=len(self.hand_points), hands_seen=len(hands),
                                eligible_hands=0)
        body = self.body(poses, face, identity, now, association_at=association_at)
        if body is None:
            self.diagnostics['current_blockers'] = [self.diagnostics['current_reason']]
            return None
        p, track, identity_key, width, center = body
        if now < self.next_at:
            self.note('cooldown')
            return None
        candidates = []
        blockers = self.diagnostics['current_blockers']

        def reject(reason):
            self.note(reason)
            if reason not in blockers:
                blockers.append(reason)

        for hand in hands:
            if len(hand) != 21 or any(not math.isfinite(v.x+v.y) for v in hand):
                reject('no_single_open_palm')
                continue
            xy = lambda i: (hand[i].x, hand[i].y)
            if any(not (0 <= hand[i].x <= 1 and 0 <= hand[i].y <= 1) for i in (0,5,9,17)):
                reject('hand_outside'); continue
            palm_width = math.dist(xy(5), xy(17))
            if not .025 <= palm_width <= .4:
                reject('hand_scale'); continue
            if hand[0].y > center[1]+.08:
                reject('hand_lowered'); continue
            extended = sum(math.dist(xy(tip), xy(0)) > 1.15*math.dist(xy(pip), xy(0))
                           for tip,pip in ((8,6),(12,10),(16,14),(20,18)))
            if extended < 3:
                reject('palm_not_open'); continue
            candidates.append(hand)
        self.diagnostics['eligible_hands'] = len(candidates)
        q = self.hand_points
        if len(candidates) != 1:
            # Only eligible palms contribute evidence. A short side-turn or
            # dropout keeps actual samples while body() still guards every frame.
            if len(candidates)>1 or (q and now>q[-1][0]+self.MAX_HAND_GAP):
                q.clear()
            self.note('no_single_open_palm')
            if not hands:
                blockers.append('no_hands')
                self.diagnostics['current_reason'] = 'no_hands'
            elif not candidates:
                self.diagnostics['current_reason'] = next(
                    (reason for reason in ('palm_not_open', 'hand_lowered', 'hand_scale',
                                          'hand_outside') if reason in blockers),
                    'no_single_open_palm')
            elif 'no_single_open_palm' not in blockers:
                blockers.append('no_single_open_palm')
            return None
        hand = candidates[0]
        x = (sum(hand[i].x for i in (5,9,17))/3-center[0])/width
        angle = math.atan2(hand[9].x-hand[0].x, hand[0].y-hand[9].y)
        if q and (now>q[-1][0]+self.MAX_HAND_GAP or abs(x-q[-1][1])>1.5):
            q.clear()
        if q:
            # atan2 wraps at +/-pi; crossing that boundary is not a full turn.
            angle = q[-1][2] + (angle-q[-1][2]+math.pi) % (2*math.pi)-math.pi
        q.append((now,x,angle))
        while q and now-q[0][0]>3.5:
            q.popleft()
        self.diagnostics['longest_run'] = max(self.diagnostics['longest_run'],len(q))
        self.note('hand_tracking')
        # Four measured extrema can establish two reversals without inventing
        # intermediate samples. Duration, excursion and span stay unchanged.
        if len(q)<self.MIN_SAMPLES or now-q[0][0]<.65:
            return None
        translate = self.reversals([v[1] for v in q],.25,.45)
        rotate = self.reversals([v[2] for v in q],.22,.45)
        if not (translate or rotate):
            self.note('motion_incomplete')
            return None
        # Naming additionally requires this actual hand to meet this body's
        # visible wrist. Unassociated/off-screen hands never borrow a face name.
        associated = any(p[i].visibility>=.5 and
            math.dist((p[i].x,p[i].y),(hand[0].x,hand[0].y))<=.18 for i in (15,16))
        self.next_at=now+4
        q.clear()
        self.note('wave_hand_rotation' if rotate else 'wave_hand_translation')
        return {'kind':'wave','event_id':secrets.token_hex(16),'source':'camera.front',
                'observed_at':now,'confidence':min(p[i].visibility for i in (0,11,12)),
                'track_id':track if associated else None,
                'identity_key':identity_key if associated else None,
                'generation':generation,'body_epoch':self.epoch}
