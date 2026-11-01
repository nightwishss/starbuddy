"""Private, bounded subprocess adapter for the opt-in Melo QNN backend."""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import threading
import time


class QnnSpeechError(RuntimeError):
    pass


class QnnTtsProcess:
    def __init__(self, python, root, models, output_root, *, timeout=27, idle_seconds=300,
                 clock=time.monotonic, shared_root=None):
        self.python, self.root, self.models = Path(python), Path(root), Path(models)
        self.output_root = Path(output_root)
        self.shared_root=Path(shared_root) if shared_root else None
        self.timeout, self.idle_seconds, self.clock = timeout, idle_seconds, clock
        self.process = self.state = None
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.retry_after = 0
        self.last = {}
        self.error_code = None
        self.failure_type = None

    def configured(self):
        return (self.python.is_file() and (self.root/'build/qnn_tts_runner').is_file()
                and (self.root/'assets/manifest.json').is_file()
                and all((self.models/(name+'.bin')).is_file()
                        for name in ('bert_wrapper', 'encoder', 'flow', 'decoder')))

    def status(self):
        # Snapshot only: no inference, startup, or blocking worker RPC.
        return {'configured': self.configured(), 'worker_alive': bool(
            self.process is not None and self.process.poll() is None),
            'retry_after_seconds': None if self.retry_after == float('inf') else round(max(0, self.retry_after-self.clock()), 2),
            'last_error': self.error_code, 'failure_type': self.failure_type,
            'last_synthesis': dict(self.last)}

    def _start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self._close()
        if self.error_code == 'cleanup_unconfirmed':
            raise QnnSpeechError('上一 NPU worker 的退出尚未确认。')
        state_root = self.shared_root/'clients' if self.shared_root else self.output_root.parent/'qnn-state'
        state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state = tempfile.TemporaryDirectory(prefix='worker-', dir=state_root)
        command=[
            str(self.python), '-B', '-m', 'gateway.speech.qnn_worker',
            '--root', str(self.root), '--models', str(self.models),
            '--state', self.state.name, '--output-root', str(self.output_root),
            '--idle-seconds', str(self.idle_seconds)]
        if self.shared_root:command.extend(['--shared-root',str(self.shared_root)])
        self.process = subprocess.Popen(command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=0)

    def _response(self):
        deadline = self.clock()+self.timeout
        while True:
            end = self.buffer.find(b'\n')
            if end >= 0:
                data = bytes(self.buffer[:end]); del self.buffer[:end+1]
                result = json.loads(data)
                if not isinstance(result, dict): raise ValueError('invalid worker response')
                return result
            if len(self.buffer) > 16384: raise ValueError('oversized worker response')
            remaining = deadline-self.clock()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                raise TimeoutError('worker deadline')
            data = os.read(self.process.stdout.fileno(), 16384)
            if not data: raise RuntimeError('worker exited')
            self.buffer.extend(data)

    def synthesize(self, text, output):
        with self.lock:
            if self.clock() < self.retry_after:
                raise QnnSpeechError('NPU 语音暂不可用，正在等待资源恢复。')
            try:
                self._start()
                request = json.dumps({'text': text, 'output': str(output)}, ensure_ascii=False)
                self.process.stdin.write((request+'\n').encode()); self.process.stdin.flush()
                result = self._response()
                if result.get('ok') is not True:
                    code = result.get('code')
                    self.error_code = code if code in {'input_limit', 'resource_busy', 'native_unavailable'} else 'worker_error'
                    raise RuntimeError(self.error_code)
                self.last = {k:result[k] for k in ('backend', 'total_ms', 'native_ms',
                    'segments', 'sample_rate', 'audio_seconds', 'context_reused') if k in result}
                self.error_code = None
            except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
                self.failure_type = type(exc).__name__
                self.error_code = self.error_code or 'worker_error'
                if self.error_code != 'input_limit': self.retry_after = self.clock()+30
                self._close()
                raise QnnSpeechError('NPU 语音合成未完成（'+self.error_code+'）。') from exc

    def _close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=2)
            for stream in (process.stdin, process.stdout):
                if stream is not None: stream.close()
        self.buffer.clear()
        # The worker confirms its owned Docker process has stopped before
        # removing this index. On an abnormal kill retain evidence for cleanup.
        if self.state is not None:
            index = Path(self.state.name)/'active-container.json'
            if index.exists():
                try:
                    if index.is_symlink() or index.stat().st_size > 4096:
                        raise ValueError('invalid owner index')
                    entry=json.loads(index.read_text())
                    if entry.get('shared'):
                        from gateway.speech.qnn_shared import exchange
                        if self.shared_root is None:raise ValueError('shared transport not configured')
                        exchange(self.shared_root,{'op':'close','owner':entry['owner']},timeout=8)
                    else:
                        from experiments.tts_acceleration.qnn_tts_context_probe import cleanup_owned_container
                        cleanup_owned_container(entry['owner'])
                    index.unlink()
                    self.state.cleanup()
                except Exception as exc:
                    self.failure_type = 'cleanup:'+type(exc).__name__
                    self.error_code = 'cleanup_unconfirmed'
                    self.retry_after = float('inf')
                    self.state._finalizer.detach()
            else:
                self.state.cleanup()
            self.state = None

    def close(self):
        with self.lock: self._close()
