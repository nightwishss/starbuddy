"""Explicit short board qualification; public prompts, aggregate metadata only.

Default: three local Qwen calls, no media capture, no sound, no cloud/history.
--contention adds a 2-second test budget reservation (not native NPU load).
--audible adds one public Output/TTS job and waits for its terminal receipt.
Never changes flags/services or clears a quarantined native lease.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def read(port, path):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=2) as response:
        return json.loads(response.read(128 * 1024))


def main():
    from control_plane import ControlPlaneClient
    from gateway.llm.client import LocalLLMClient
    from output_service import OutputClient
    from telemetry_service import TelemetryClient
    parser = argparse.ArgumentParser()
    parser.add_argument('--contention', action='store_true')
    parser.add_argument('--audible', action='store_true')
    args = parser.parse_args()
    control, output = ControlPlaneClient(timeout=.5), OutputClient(timeout=2)
    status = output.system_status()
    if (not status.get('ready') or status.get('current') or status.get('queued') != 0
            or status.get('output_quarantined') or status.get('tts_quarantined')
            or read(8765, '/api/interaction/state').get('phase') != 'idle'):
        raise RuntimeError('interactive_work_in_progress')
    if any(lease['resources'].get('npu_compute', 0) for lease in control.status()['leases']):
        raise RuntimeError('compute_in_progress')
    def pose():
        value = read(8791, '/status')
        return {k: value.get(k) for k in ('sequence', 'last_frame_age_ms', 'ready', 'resource_admission')}
    before = pose()
    rows = []
    for _ in range(3):
        started = time.monotonic()
        answer = LocalLLMClient(timeout=25).chat('软件验收，请仅回答一句：短暂休息有助于保持专注。',
                                                max_completion_tokens=48, temperature=.1)
        rows.append(round((time.monotonic() - started) * 1000, 2))
        if not answer.strip():
            raise RuntimeError('empty_completion')
    record = {'local_qwen_ms': rows, 'median_ms': statistics.median(rows),
              'pose_before': before, 'pose_after_qwen': pose(), 'npu_utilization_measured': False}
    if args.contention:
        lease = control.acquire('probe:posture-resource', {'npu_compute': 100},
                                'probe.contention', priority=80, ttl_seconds=3)
        try:
            time.sleep(.5)
            first = pose()
            time.sleep(.5)
            second = pose()
            if first['sequence'] != second['sequence']:
                raise RuntimeError('pose_did_not_yield')
            if second['resource_admission']['reason'] != 'foreground_compute':
                raise RuntimeError('budget_deferral_not_visible')
        finally:
            control.release(lease['lease_id'], 'probe:posture-resource')
        time.sleep(.8)
        recovered = pose()
        if recovered['sequence'] <= second['sequence']:
            raise RuntimeError('pose_did_not_resume')
        record['budget_replay'] = {'yielded': True, 'resumed': True,
                                   'frames_after_release': recovered['sequence'] - second['sequence']}
    if args.audible:
        job = output.say('这是一条资源调度验收提示。', 'session:resource-acceptance',
                         category='notification', expires_in_seconds=45)
        deadline = time.monotonic() + 50
        try:
            while time.monotonic() < deadline:
                receipt = output.status(job['job_id'])
                if receipt.get('ended_at') is not None:
                    break
                time.sleep(.2)
            else:
                raise RuntimeError('output_deadline')
            if receipt['state'] != 'completed':
                raise RuntimeError('output_not_completed')
            record['tts'] = {'state': receipt['state'], 'timing_ms': receipt.get('timing_ms'),
                              'acoustic_onset_measured': False}
        finally:
            if output.status(job['job_id']).get('ended_at') is None:
                output.cancel(job['job_id'])
    telemetry = TelemetryClient(timeout=2).latest()
    record['resources'] = {
        'cpu_percent': telemetry['system']['cpu']['percent'],
        'available_memory_bytes': telemetry['system']['memory']['available_bytes'],
        'context_cpu_percent': telemetry['applications'].get('vq-perception-context', {}).get('cpu_percent'),
        'pose_cpu_percent': telemetry['applications'].get('vq-pose-npu', {}).get('cpu_percent'),
    }
    record['pose_final'] = pose()
    record['compute_used_after'] = control.status()['used']['npu_compute']
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
