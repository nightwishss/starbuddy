"""UI-leased gesture worker and guarded Output delivery. Off after reload/leave/restart."""
import json
import re
import secrets
import selectors
import subprocess
import threading
import time
from pathlib import Path

from perception_context_service.client import PerceptionContextClient
from output_service.client import OutputClient
from runtime.client import RuntimeClient
from sdk.speech import SpeechClient
from media_service import MediaClient


class WaveController:
    # QNN unloads after 300s idle. Prepare once on explicit enable, then at
    # most once per 240s while this UI lease remains active AND output is idle.
    # Only this public phrase is generated; no names/profiles/audio are cached.
    SPEECH_WARM_INTERVAL = 240
    SPEECH_WARM_TEXT = '你好呀。'
    STOP_REASONS = frozenset({'manual_stop', 'page_hidden', 'page_left', 'navigation',
                             'user_interaction', 'heartbeat_failed'})

    def __init__(self, app):
        self.app = app
        self.lock = threading.RLock()
        self.submit_lock = threading.Lock()
        self.enabled = False
        self.owner = None
        self.seen = 0
        self.generation = 0
        self.process = None
        self.thread = None
        self.ready = False
        self.reason = 'off'
        self.last_stop_reason = 'service_started'
        self.stopped_at = None
        self.worker_exit_code = None
        self.frames = 0
        self.latency_ms = None
        self.output_job = None
        self.greeting = None
        self.greeting_revision = 0
        self.decision_count = 0
        self.last_decision = 'none'
        self.person_present, self.observed_at, self.body_epoch = False, 0, 0
        self.detector = {}
        self.speech_state = 'not_prepared'
        self.speech_prepared_at = 0
        self.speech_prepare_ms = None
        self.speech_next_at = 0
        self.speech_inflight = False
        self.speech_prepare_thread = None
        self.last_response_timing = None

    def status(self):
        with self.lock:
            if self.greeting and time.monotonic()-self.greeting['at'] >= 12:
                self.greeting = None
            enabled = self.enabled and time.monotonic()-self.seen < 8
            reason = 'lease_expired' if self.enabled and not enabled else self.reason
            quiet = self.app.quiet
            return {'enabled': enabled, 'ready': enabled and self.ready,
                    'generation': self.generation, 'reason': reason,
                    'last_stop_reason': self.last_stop_reason, 'stopped_at': self.stopped_at,
                    'worker_exit_code': self.worker_exit_code,
                    'heartbeat_age_ms': round(max(0, time.monotonic()-self.seen)*1000) if self.seen else None,
                    'frames': self.frames, 'latency_ms': self.latency_ms,
                    'person_present': self.person_present and enabled,
                    'observed_at': self.observed_at, 'body_epoch': self.body_epoch,
                    'frame_age_ms': round(max(0, time.time()-self.observed_at)*1000) if self.observed_at else None,
                    'detector': dict(self.detector),
                    'arrival_enabled': False,
                    'visual_response_policy': 'wave_only',
                    'quiet': quiet.state in {'playing', 'paused'} or quiet.quarantined,
                    'output_job': self.output_job,
                    # Silent background recognition is compute work, not a user
                    # turn. Explicit visual questions already hold app.lock.
                    'background_visual_busy': self.app.scene_lock.locked(),
                    'user_busy': any(v.locked() for v in (self.app.lock, self.app.voice_lock, self.app.touch_lock)),
                    'turn_active': self.app.turn_manager.snapshot()['turn_id'] is not None,
                    'decision_count': self.decision_count, 'last_decision': self.last_decision,
                    # Software receipt timings only, not acoustic onset or identity.
                    'response_timing': dict(self.last_response_timing) if self.last_response_timing else None,
                    'speech_preparation': {
                        'state': self.speech_state if enabled else 'not_prepared',
                        'age_ms': round((time.monotonic()-self.speech_prepared_at)*1000)
                                  if enabled and self.speech_prepared_at else None,
                        'duration_ms': self.speech_prepare_ms if enabled else None},
                    'raw_media_retained': False, 'microphone_started': False}

    def _stop_locked(self, reason):
        self.enabled = self.ready = False
        self.reason = self.last_stop_reason = reason
        self.stopped_at = time.time()
        self.generation += 1

    def control(self, action, owner, reason=None):
        if action not in {'start', 'heartbeat', 'stop'} or not isinstance(owner, str) or not re.fullmatch(r'[a-f0-9]{32}', owner):
            raise ValueError('invalid wave control')
        if reason is not None and (action != 'stop' or reason not in self.STOP_REASONS):
            raise ValueError('invalid stop reason')
        with self.lock:
            if self.enabled and time.monotonic()-self.seen >= 8:
                self._stop_locked('lease_expired')
            if self.enabled and self.owner != owner and time.monotonic()-self.seen < 8:
                raise ValueError('another window owns visual interaction')
            if action == 'stop' and self.owner is not None and self.owner != owner:
                raise ValueError('another window owns visual interaction')
            if action == 'start':
                if self.thread and self.thread.is_alive() and not self.enabled:
                    raise ValueError('visual worker is stopping; try again shortly')
                if not self.enabled:
                    self.generation += 1
                    self.enabled, self.ready, self.reason = True, False, 'loading'
                    self.worker_exit_code = None
                    self.owner, self.frames, self.greeting = owner, 0, None
                    self.seen = time.monotonic()
                    self.speech_state = ('cleanup_unconfirmed' if self.speech_next_at == float('inf')
                                         else 'not_prepared')
                    self.speech_prepared_at = 0
                    self.speech_prepare_ms = None
                    # An unconfirmed media cleanup disables preparation for
                    # this controller lifetime, including a manual re-enable.
                    if self.speech_next_at != float('inf'):
                        self.speech_next_at = 0
                    self.thread = threading.Thread(target=self._run, args=(self.generation,), daemon=True,
                                                   name='vq-wave-fusion')
                    self.thread.start()
            elif action == 'stop':
                # A late pagehide must not overwrite the actual worker/lease failure.
                if self.enabled:
                    self._stop_locked(reason or 'manual_stop')
            elif not self.enabled or self.owner != owner:
                return self.status()
            self.seen = time.monotonic()
            result = self.status()
            if self.greeting and time.monotonic()-self.greeting['at'] < 12:
                result['greeting'] = {k: v for k, v in self.greeting.items() if k != 'at'}
        if action == 'stop':
            self._cancel_owned()
        return result

    def _cancel_owned(self):
        with self.lock:
            job = self.output_job
        if job and self.app.output is not None:
            try:
                OutputClient(self.app.output.socket_path, timeout=1).cancel(job)
            except Exception:
                pass  # Guard is invalidated even if native cancellation is unavailable.

    def _active(self, generation):
        with self.lock:
            return self.enabled and self.generation == generation and time.monotonic()-self.seen < 8

    def _speech_idle(self):
        app = self.app
        return (app.output is not None and self.output_job is None
                and app.active_playback_id is None
                and not any(v.locked() for v in (app.lock, app.voice_lock, app.touch_lock))
                and app.turn_manager.snapshot()['turn_id'] is None
                and app.quiet.state not in {'playing', 'paused'} and not app.quiet.quarantined)

    def _prepare_speech(self, generation):
        # Never hold the controller lock across IPC/native inference: heartbeat,
        # pagehide, stop and status must stay responsive during a cold start.
        with self.lock:
            if (not self._active(generation) or self.speech_inflight
                    or time.monotonic() < self.speech_next_at):
                return False
            self.speech_inflight = True
            self.speech_state = 'preparing'
            self.speech_next_at = time.monotonic() + self.SPEECH_WARM_INTERVAL
        media_id = None
        cleaned = True
        success = False
        state = 'failed'
        started = time.monotonic()
        try:
            if not self._speech_idle():
                state = 'deferred'
                return False
            status = OutputClient(self.app.output.socket_path, timeout=2).system_status()
            if (status.get('ready') is not True or status.get('current') is not None
                    or type(status.get('queued')) is not int or status['queued'] != 0):
                state = 'deferred'
                return False
            # Recheck the lease and explicit user intent after the IPC boundary.
            with self.lock:
                if not self._active(generation) or not self._speech_idle():
                    state = 'deferred'
                    return False
            result = SpeechClient(RuntimeClient(timeout=35)).synthesize(
                self.SPEECH_WARM_TEXT, 'session:wave-prewarm')
            candidate_id = result.get('media_id')
            if not isinstance(candidate_id, str) or not re.fullmatch(r'generated_[A-Za-z0-9_-]{20,64}', candidate_id):
                raise ValueError('invalid preparation media handle')
            media_id = candidate_id
            success = True
            state = 'ready'
        except Exception:
            pass
        finally:
            if media_id is not None:
                try:
                    receipt = MediaClient(timeout=3).call(
                        'release', media_id=media_id, owner='session:wave-prewarm')
                    if (not isinstance(receipt, dict) or receipt.get('released') is not True
                            or receipt.get('media_id') != media_id):
                        raise ValueError('unconfirmed preparation cleanup')
                except Exception:
                    cleaned = False
            with self.lock:
                self.speech_inflight = False
                if not cleaned:
                    self.speech_next_at = float('inf')
                    self.speech_state = 'cleanup_unconfirmed'
                if self.generation == generation and self.enabled:
                    self.speech_state = state if cleaned else 'cleanup_unconfirmed'
                    self.speech_prepare_ms = round((time.monotonic()-started)*1000, 2)
                    if success and cleaned:
                        self.speech_prepared_at = time.monotonic()
                    # An idle gate is cheap to recheck, but a failed/unknown
                    # model outcome never causes an immediate native retry.
                    if state == 'deferred' and cleaned:
                        self.speech_next_at = time.monotonic() + 15
        return success and cleaned

    def _maybe_prepare_speech(self, generation):
        with self.lock:
            if (not self._active(generation) or self.speech_inflight
                    or time.monotonic() < self.speech_next_at
                    or self.speech_prepare_thread is not None and self.speech_prepare_thread.is_alive()):
                return
            self.speech_prepare_thread = threading.Thread(
                target=self._prepare_speech, args=(generation,), daemon=True, name='vq-wave-speech-prepare')
            self.speech_prepare_thread.start()

    def _run(self, generation):
        process = None
        selector = None
        stop_reason = 'worker_exited'
        try:
            self._prepare_speech(generation)
            # Stopping during the cold preparation must not start a camera later.
            root = Path(__file__).resolve().parents[2]
            # Only the short spawn belongs to this atomic lease check. No model
            # work or remote calls run under the lock. A concurrent stop either
            # wins here (no spawn) or terminates this already-started worker.
            with self.lock:
                if not self._active(generation):
                    if self.enabled and self.generation == generation and time.monotonic()-self.seen >= 8:
                        stop_reason = 'lease_expired'
                    return
                process = subprocess.Popen([
                    '/home/arduino/.local/share/vq-pose-lab/venv/bin/python', '-B', '-u',
                    '-m', 'gateway.display.wave_worker', '--generation', str(generation)],
                    cwd=root, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    bufsize=0)
                self.process = process
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            buffer = b''
            last_status = time.monotonic()
            while self._active(generation) and process.poll() is None:
                self._maybe_prepare_speech(generation)
                if time.monotonic()-last_status > 15:
                    raise TimeoutError('worker_timeout')
                for key, _ in selector.select(.2):
                    chunk = key.fileobj.read(4096)
                    if not chunk:
                        break
                    buffer += chunk
                    if len(buffer) > 16384:
                        raise ValueError('worker_invalid_output')
                    while b'\n' in buffer:
                        raw, buffer = buffer.split(b'\n', 1)
                        value = json.loads(raw)
                        if not isinstance(value, dict) or value.get('type') not in {'status', 'event'}:
                            raise ValueError('worker_invalid_output')
                        if value.get('type') == 'status':
                            last_status = time.monotonic()
                            with self.lock:
                                if not self._active(generation):
                                    break
                                self.ready = value.get('ready') is True
                                self.reason = 'observing' if self.ready else 'camera_unavailable'
                                self.frames = value.get('frames', 0)
                                self.latency_ms = value.get('latency_ms')
                                self.person_present = value.get('person_present') is True
                                self.observed_at = value.get('observed_at', 0)
                                self.body_epoch = value.get('body_epoch', 0)
                                self.detector = value.get('detector') or {}
                        elif value.get('type') == 'event' and self._active(generation):
                            self._submit(value['event'])
            with self.lock:
                if time.monotonic()-self.seen >= 8:
                    stop_reason = 'lease_expired'
                elif process.poll() is not None:
                    self.worker_exit_code = process.returncode
                    stop_reason = 'worker_exited' if process.returncode == 0 else 'worker_crashed'
        except TimeoutError:
            stop_reason = 'worker_timeout'
        except (ValueError, KeyError, TypeError):
            stop_reason = 'worker_invalid_output'
        except OSError:
            stop_reason = 'worker_start_failed' if process is None else 'worker_io_error'
        except Exception:
            stop_reason = 'worker_error'
        finally:
            if selector is not None:
                selector.close()
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
                process.stdout.close()
            with self.lock:
                self.process = None
                if self.enabled and self.generation == generation:
                    self._stop_locked(stop_reason)
            self._cancel_owned()

    def arrival(self):
        # Retain bounded arrival context in Fusion without requesting speech.
        with self.lock:
            generation = self.generation
        self._submit({'kind': 'arrival', 'event_id': secrets.token_hex(16),
                      'source': 'camera.environment', 'observed_at': time.time(), 'confidence': .9,
                      'track_id': None, 'identity_key': None, 'generation': generation, 'body_epoch': None})

    def _submit(self, event):
        if not self.submit_lock.acquire(blocking=False):
            return
        try:
            app = self.app
            decision = PerceptionContextClient(timeout=2).call('fusion.greeting.proposal', event=event)
            with self.lock:
                self.decision_count += 1
                self.last_decision = decision['reason']
            if event.get('kind') != 'wave' or decision.get('action') != 'greet' or app.output is None:
                return
            # The Output arbiter revalidates fusion at dequeue AND immediately
            # before playback. No unguarded synthesis/player fallback or retry.
            output = OutputClient(app.output.socket_path, timeout=2)
            result = output.say(decision['text'], 'session:wave-greeting', category='notification',
                                volume_percent=60, expires_in_seconds=decision['expires_in_seconds'],
                                guard=decision['guard'], trace_id='wave-'+event['event_id'],
                                speed=app.preferences.snapshot()['values']['speed'])
            job = result['job_id']
            with self.lock:
                self.output_job = job
            with app.playback_lock:
                if app.active_playback_id is not None:
                    output.cancel(job)
                else:
                    app.active_playback_id = job
            threading.Thread(target=self._watch_job, args=(job, decision['text'], event['generation'], event['observed_at']),
                             daemon=True, name='vq-wave-receipt').start()
        except Exception:
            with self.lock:
                self.last_decision = 'fusion_or_output_unavailable'
                # An unknown submit outcome must not be retried or played later.
                self._stop_locked('fusion_or_output_unavailable')
        finally:
            self.submit_lock.release()

    def _watch_job(self, job, text, generation, observed_at=None):
        output = OutputClient(self.app.output.socket_path, timeout=1)
        shown = False
        terminal = False
        for _ in range(60):
            try:
                result = output.status(job)
                state = result.get('state')
                timings = result.get('timing_ms') or {}
                if 'to_playback' in timings and type(observed_at) in (int, float):
                    dispatch_ms = (result['created_at']-observed_at)*1000
                    with self.lock:
                        if generation == self.generation:
                            self.last_response_timing = {
                                'job_id': job, 'state': state,
                                'gesture_to_submit_ms': round(max(0, dispatch_ms), 2),
                                'gesture_to_player_receipt_ms': round(max(0, dispatch_ms)+timings['to_playback'], 2),
                                'output_ms': {k: timings[k] for k in
                                              ('queue', 'synthesis', 'playback_submit', 'to_playback', 'playback_lead_in')
                                              if k in timings}}
                if state == 'playing' and not shown:
                    with self.lock:
                        if generation != self.generation or self.app.active_playback_id != job:
                            output.cancel(job)
                        else:
                            self.greeting_revision += 1
                            self.greeting = {'text': text, 'revision': self.greeting_revision,
                                             'at': time.monotonic()}
                            self.last_decision = 'playing'
                            if self.app.turn_manager.snapshot()['turn_id'] is None:
                                self.app.state.update({'phase': 'speaking', 'visual_style': 'curious',
                                                       'primary_text': text, 'secondary_text': '视觉问候 · 不会自动收音',
                                                       'producer': 'vision'})
                    shown = True
                if state in {'completed', 'cancelled', 'interrupted', 'expired', 'failed'}:
                    with self.lock:
                        self.last_decision = 'output_'+state
                    terminal = True
                    break
            except Exception:
                pass
            time.sleep(.5)
        if not terminal:
            try:
                output.cancel(job)
            except Exception:
                pass
            # Keep pending ownership: unknown native cleanup is not idle.
            with self.lock:
                self.last_decision = 'output_cleanup_unknown'
            return
        with self.app.playback_lock:
            if self.app.active_playback_id == job:
                self.app.active_playback_id = None
                if self.app.turn_manager.snapshot()['turn_id'] is None:
                    self.app.state.update({'phase': 'idle', 'visual_style': 'gentle',
                                           'primary_text': '我在呢。', 'secondary_text': '', 'producer': 'vision'})
        with self.lock:
            if self.output_job == job:
                self.output_job = None

    def close(self):
        with self.lock:
            if self.enabled:
                self._stop_locked('service_shutdown')
        self._cancel_owned()
        if self.thread:
            self.thread.join(timeout=4)
