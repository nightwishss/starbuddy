"""Opt-in Runtime child: persistent frontend, bounded reusable QNN contexts.

No service stopping, model downloads, automatic CPU fallback or audio output.
Media and Output remain the sole downstream playback path. Private tensors are
discarded after every graph. Only bounded timing metadata returns to Runtime.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import select
import signal
import sys
import tempfile
import time
import uuid


def validate_request(value, output_root):
    if not isinstance(value, dict) or set(value) != {'text', 'output'}:
        raise ValueError('invalid request')
    text = value['text']
    if not isinstance(text, str) or not text.strip() or len(text) > 2020:
        raise ValueError('input_limit')
    output = Path(value['output'])
    if (not output.is_absolute() or output.parent.resolve() != output_root.resolve()
            or not re.fullmatch(r'runtime-tts-[A-Za-z0-9_-]{20,64}\.wav', output.name)
            or output.exists() or output.is_symlink()):
        raise ValueError('invalid output')
    return text, output


def segments(text):
    from gateway.speech.text import split_for_speech
    parts = split_for_speech(text, first_limit=64, next_limit=64, max_segments=16)
    if not parts: raise ValueError('input_limit')
    return parts


class Resident:
    """Own exactly one native worker and renew its RAM lease while alive."""
    def __init__(self, args, control, frontend, client_factory, synthesize, write_wav,
                 clock=time.monotonic):
        self.args, self.control, self.frontend = args, control, frontend
        self.client_factory, self.synthesize, self.write_wav = client_factory, synthesize, write_wav
        self.clock, self.client, self.lease, self.directory = clock, None, None, None
        self.began = self.renewed = 0
        self.owner = 'tts-qnn-runtime:'+uuid.uuid4().hex[:16]

    def tick(self):
        if self.lease and self.clock()-self.renewed >= 5:
            heartbeat=getattr(self.client,'heartbeat',None)
            if heartbeat:heartbeat()
            self.control.renew(self.lease['lease_id'], self.owner, ttl_seconds=60)
            self.renewed = self.clock()

    def ensure(self):
        reused = self.client is not None
        if self.client and (self.client.count+42 > 256 or self.clock()-self.began > 500):
            self.close(); reused = False
        if self.client is None:
            self.lease = self.control.acquire(self.owner, {'ram_mb': 1000},
                'speech.qnn.resident', priority=80, ttl_seconds=60)
            self.renewed = self.clock()
            self.directory = tempfile.TemporaryDirectory(prefix='context-', dir=self.args.state)
            self.client = self.client_factory(self.args.root/'build/qnn_tts_runner', self.args.models,
                Path(self.directory.name)/'worker', timeout_seconds=600,
                max_commands=256, discard_tensors=True)
            index = Path(self.args.state)/'active-container.json'
            with index.open('x') as stream:
                os.fchmod(stream.fileno(), 0o600)
                json.dump({'owner': self.client.owner, 'shared': bool(getattr(self.args,'shared_root',None))}, stream); stream.flush(); os.fsync(stream.fileno())
            self.client.start(); self.began = self.clock()
        self.tick()
        return reused

    def run(self, text, output):
        import numpy as np
        parts = segments(text)
        began = self.clock()
        compute = self.control.acquire(self.owner, {'npu_compute': 100},
            'speech.qnn.synthesize', priority=80, ttl_seconds=60)
        arrays, native_ms, reused = [], 0, True
        release_compute = True
        try:
            for part in parts:
                reused = self.ensure() and reused
                self.tick()
                first = len(self.client.stats)
                audio, _ = self.synthesize(self.client, self.frontend, part)
                native_ms += sum(x['execute_ms'] for x in self.client.stats[first:])
                arrays.append(audio)
            audio = np.concatenate(arrays)
            self.write_wav(output, audio)
            return {'ok': True, 'backend': 'melo-qnn-htp', 'total_ms': round((self.clock()-began)*1000, 2),
                'native_ms': round(native_ms, 2), 'segments': len(parts), 'sample_rate': 44100,
                'audio_seconds': len(audio)/44100, 'context_reused': reused}
        except BaseException:
            try: self.close()
            except BaseException:
                release_compute = False
                raise
            raise
        finally:
            if release_compute: self.control.release(compute['lease_id'], self.owner)

    def close(self):
        if self.client:
            closed = self.client.close()
            if closed.get('context_release_confirmed',closed.get('container_exit_confirmed')) is not True:
                raise RuntimeError('cleanup unconfirmed')
            self.client = None
            (Path(self.args.state)/'active-container.json').unlink(missing_ok=True)
        if self.lease:
            self.control.release(self.lease['lease_id'], self.owner)
            self.lease = None
        if self.directory:
            self.directory.cleanup(); self.directory = None


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'models', 'state', 'output-root'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--idle-seconds', type=int, default=300)
    parser.add_argument('--shared-root', type=Path)
    args = parser.parse_args()
    if not 10 <= args.idle_seconds <= 600: raise ValueError('idle limit')
    # Private protocol fd remains separate from noisy upstream text libraries.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    os.umask(0o077)
    source = Path(__file__).resolve().parents[2]/'experiments/tts_acceleration'
    sys.path.insert(0, str(source))
    from qnn_tts_chain import QnnChainClient, synthesize_chain, write_public_wav
    if args.shared_root:
        from functools import partial
        from gateway.speech.qnn_shared_client import SharedQnnClient
        QnnChainClient=partial(SharedQnnClient,shared_root=args.shared_root)
    from qnn_tts_chain_probe import validate_assets, validate_models
    from melo_npu_frontend import MeloNpuFrontend
    from control_plane.client import ControlPlaneClient
    def interrupted(*_): raise InterruptedError('worker stopping')
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGALRM):
        signal.signal(sig, interrupted)
    resident = None
    try:
        validate_assets(args.root/'assets'); validate_models(args.models, args.root/'build/qnn_tts_runner')
        frontend = MeloNpuFrontend(args.root/'assets/chinese-config.json',
            args.root/'assets/tokenizer.json', args.root/'assets', args.root/'assets/nltk-data')
        resident = Resident(args, ControlPlaneClient(timeout=1), frontend,
                            QnnChainClient, synthesize_chain, write_public_wav)
        last, buffer = time.monotonic(), bytearray()
        while time.monotonic()-last < args.idle_seconds:
            resident.tick()
            if not select.select([sys.stdin], [], [], 1)[0]: continue
            data = os.read(sys.stdin.fileno(), 16384)
            if not data: break
            buffer.extend(data)
            if len(buffer) > 16384: raise ValueError('request too large')
            if b'\n' not in buffer: continue
            line, _, tail = buffer.partition(b'\n'); buffer = bytearray(tail)
            if buffer: raise ValueError('pipelining not supported')
            output = None
            try:
                signal.setitimer(signal.ITIMER_REAL, 24)
                text, output = validate_request(json.loads(line), args.output_root)
                result = resident.run(text, output)
            except Exception as exc:
                if output: output.unlink(missing_ok=True)
                code = 'input_limit' if isinstance(exc, ValueError) else 'resource_busy' if getattr(exc, 'code', '') == 'resource_busy' else 'native_unavailable'
                protocol.write(json.dumps({'ok': False, 'code': code})+'\n')
                break
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            protocol.write(json.dumps(result)+'\n')
            last = time.monotonic()
    except (InterruptedError, OSError, ValueError, RuntimeError):
        pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP): signal.signal(sig, signal.SIG_IGN)
        if resident: resident.close()
        protocol.close()


if __name__ == '__main__':
    main()
