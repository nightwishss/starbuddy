"""Finite QNN/HTP TTS subgraph smoke test, NOT text-to-speech acceptance.

Only synthetic latent tensors. Uses a CPU/RAM/NPU lease and never plays audio,
replaces production models or permits a CPU execution backend fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid


def tensor_fixture(info, name):
    import numpy as np
    shape = info['shape']
    if not shape or any(type(v) is not int or not 1 <= v <= 4096 for v in shape):
        raise ValueError('invalid static shape')
    if math.prod(shape) > 2_000_000:
        raise ValueError('fixture size exceeds bound')
    dtype = info['dtype']
    if dtype == 'uint16':
        encoding = info.get('quantization_parameters', {})
        zero = encoding.get('zero_point')
        scale = encoding.get('scale')
        if (type(zero) is not int or not 0 <= zero <= 65535
                or type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0):
            raise ValueError('missing uint16 encoding; never assume FP16')
        return np.full(shape, zero, dtype='<u2')
    if dtype == 'int32':
        return np.full(shape, 1 if name in {'x_lengths', 'attention_mask'} else 0, dtype='<i4')
    if dtype == 'float32':
        value = {'length_scale': 1.0, 'noise_scale_w': .8, 'sdp_ratio': .2}.get(name, 0.0)
        return np.full(shape, value, dtype='<f4')
    raise ValueError('unsupported metadata dtype')


def native_output_path(root, index, name):
    """QAIRT 2.45 names native files *_native.raw; retain older layout support."""
    if not name.replace('_', '').isalnum() or type(index) is not int or index < 0:
        raise ValueError('unsafe output identity')
    directory = root / 'outputs' / ('Result_' + str(index))
    paths = [directory / (name + '_native.raw'), directory / (name + '.raw')]
    existing = [path for path in paths if path.is_file()]
    if len(existing) != 1:
        raise ValueError('missing or ambiguous expected native QNN output')
    raw = existing[0]
    if raw.is_symlink() or not raw.resolve().is_relative_to((root / 'outputs').resolve()):
        raise ValueError('output escapes the owned experiment directory')
    return raw


def require_idle(output):
    status = output.system_status()
    if not status.get('ready') or status.get('current') or status.get('queued') or status.get('output_quarantined'):
        raise RuntimeError('foreground Output busy/unavailable; test refused')


def container_stopped(owner):
    """Only inspect our fixed-prefix UUID container; daemon errors are unknown."""
    suffix = owner.removeprefix('tts-qnn-smoke-')
    if not owner.startswith('tts-qnn-smoke-') or len(suffix) != 12 or any(c not in '0123456789abcdef' for c in suffix):
        raise ValueError('not an owned experiment container')
    result = subprocess.run(['docker', 'container', 'inspect', '--format', '{{json .State}}', owner],
                            capture_output=True, text=True, timeout=4)
    if result.returncode != 0:
        missing = any(message in result.stderr for message in (
            'No such container: ' + owner, 'No such object: ' + owner))
        if result.returncode == 1 and missing:
            return True
        raise RuntimeError('owned container exit unknown: Docker unavailable/inspect failed')
    state = json.loads(result.stdout)
    if type(state.get('Running')) is not bool or type(state.get('Restarting')) is not bool:
        raise ValueError('invalid Docker state')
    return not state['Running'] and not state['Restarting']


def cleanup_owned_container(owner):
    if container_stopped(owner):
        return True
    # No global stop/prune. A stop racing normal --rm exit is resolved by inspect.
    subprocess.run(['docker', 'stop', '--time', '2', owner],
                   capture_output=True, text=True, timeout=8)
    if not container_stopped(owner):
        raise RuntimeError('owned container still running; lease release refused')
    return True


def interrupted(signum, _frame):
    raise InterruptedError(f'owned NPU probe interrupted by signal {signum}')


def select_container_runtime(command, platform_runtime=False):
    """Try installed vendor HTP libs without replacing any production files."""
    selected = list(command)
    if platform_runtime:
        selected[selected.index('LD_LIBRARY_PATH=/vq-sdk:/usr/lib')] = 'LD_LIBRARY_PATH=/usr/lib:/vq-sdk'
        # Single-directory control matches the working Arduino/LiteRT process.
        selected[selected.index('ADSP_LIBRARY_PATH=/vq-sdk;/vq-platform-dsp;/usr/lib/rfsa/adsp;/usr/lib/dsp/cdsp')] = 'ADSP_LIBRARY_PATH=/usr/lib/rfsa/adsp'
        selected[selected.index('--backend') + 1] = '/usr/lib/libQnnHtp.so'
    return selected


def instrument_container(command):
    """Trace only our synthetic probe child, never attach to production."""
    traced = list(command)
    index = traced.index('/vq-sdk/qnn-net-run')
    traced[index:index] = ['/usr/bin/strace', '-f', '-e', 'trace=openat,access,readlink,ioctl',
                           '-s', '160', '-o', '/vq-run/syscalls.log',
                           '/usr/bin/env', 'LD_DEBUG=libs']
    index = traced.index('--entrypoint')
    traced[index:index] = ['--ulimit', 'fsize=8388608:8388608']
    traced += ['--log_level', 'verbose']
    return traced


def direct_platform_view(command):
    """Reuse already prepared host BSP read-only; no initialization writes."""
    selected = list(command)
    selected[selected.index('/usr/share/qcom:/run/host-qcom:ro')] = '/usr/share/qcom:/usr/share/qcom:ro'
    selected[selected.index('--entrypoint') + 1] = '/usr/bin/env'
    return selected


def container_environment(command):
    # Only Docker's flags, not strace's later '-e trace=...' child option.
    prefix = command[:command.index('--entrypoint')]
    return {prefix[index + 1].split('=', 1)[0]: prefix[index + 1].split('=', 1)[1]
            for index, option in enumerate(prefix) if option == '-e'}


def unsigned_device_config():
    # Explicit ordinary unsigned PD on the already allowed nonsecure CDSP.
    # No secure device, privilege escalation or firmware change is requested.
    return {'devices': [{'device_id': 0, 'soc_id': 82, 'dsp_arch': 'v75',
                         'pd_session': 'unsigned'}]}


def litert_control_source():
    return '''import json,time,pathlib
import numpy as np
from ai_edge_litert.interpreter import Interpreter,OpResolverType
from utils.tf import load_qnn_delegate
delegates=load_qnn_delegate(delegate_options={'backend_type':'htp', 'htp_performance_mode':'2', 'htp_precision':'1', 'log_level':'1'})
if not delegates: raise RuntimeError('QNN delegate unavailable; CPU control refused')
start=time.monotonic()
model=Interpreter('/vq-pose-control/pose_landmarks_detector.tflite', experimental_delegates=delegates, num_threads=1, experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
model.allocate_tensors()
delegated=sum(op['op_name']=='DELEGATE' for op in model._get_ops_details())
if not delegated: raise RuntimeError('QNN did not delegate any nodes; CPU control refused')
init_ms=(time.monotonic()-start)*1000
info=model.get_input_details()[0]
if int(np.prod(info['shape']))>2000000: raise ValueError('oversized control tensor')
model.set_tensor(info['index'],np.zeros(info['shape'],dtype=info['dtype']))
start=time.monotonic();model.invoke();elapsed=(time.monotonic()-start)*1000
outputs=[]
for info in model.get_output_details():
 values=model.get_tensor(info['index'])
 if not np.isfinite(values).all(): raise RuntimeError('nonfinite control output')
 outputs.append({'shape':list(map(int,values.shape)), 'dtype':str(values.dtype), 'finite':True})
pathlib.Path('/vq-run/litert-control.json').write_text(json.dumps({'backend':'LiteRT QNN HTP', 'synthetic_zero_input':True, 'default_cpu_delegates_disabled':True, 'delegate_nodes':delegated,'initialization_ms':init_ms,'invoke_ms':elapsed,'outputs':outputs}))
'''


def run(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from output_service.client import OutputClient
    from control_plane.client import ControlPlaneClient
    import numpy as np

    sdk, model = args.sdk_dir.resolve(), args.model_dir.resolve()
    dsp_extra = args.dsp_extra_dir.resolve() if args.dsp_extra_dir else None
    if args.platform_runtime and (not args.container or dsp_extra):
        raise ValueError('platform runtime requires an isolated container without SDK DSP extra')
    if args.diagnostic and not args.container:
        raise ValueError('diagnostic tracing requires our isolated container')
    if args.direct_platform and not args.container:
        raise ValueError('direct read-only platform view requires our isolated container')
    if args.unsigned_pd and (not args.container or not args.platform_runtime):
        raise ValueError('unsigned PD control requires the isolated installed vendor runtime')
    if args.litert_control and (not args.container or not args.platform_runtime or args.unsigned_pd):
        raise ValueError('LiteRT control requires isolated vendor runtime, without net-run extensions')
    ioctl_trace_lib = args.ioctl_trace_lib.resolve() if args.ioctl_trace_lib else None
    if ioctl_trace_lib and (not args.container or not ioctl_trace_lib.is_file()
                            or ioctl_trace_lib.stat().st_size > 256 * 1024):
        raise ValueError('ioctl metadata shim must be a small isolated-container diagnostic library')
    metadata = json.loads((model / 'metadata.json').read_text())
    if not metadata['tool_versions']['qairt'].startswith('2.45.0.'):
        raise ValueError('model is not for the pinned QAIRT 2.45 runtime')
    attributes = metadata['chipset_attributes']
    if attributes.get('soc_model') != 82 or attributes.get('htp_version') != 75:
        raise ValueError('not the expected VENTUNO Q/IQ8275 HTP75 package')
    sdk_manifest = json.loads((sdk / 'manifest.json').read_text())
    if sdk_manifest.get('qairt_version') != '2.45.0.260326':
        raise ValueError('SDK version mismatch')
    for item in sdk_manifest['files']:
        if hashlib.sha256((sdk / item['file']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('SDK file integrity mismatch')
    if dsp_extra:
        extra_manifest = json.loads((dsp_extra / 'manifest.json').read_text())
        if extra_manifest.get('qairt_version') != sdk_manifest['qairt_version']:
            raise ValueError('DSP extra version mismatch')
        for item in extra_manifest['files']:
            if hashlib.sha256((dsp_extra / item['file']).read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('DSP extra integrity mismatch')
    component = args.component + '.bin'
    contract = metadata['model_files'][component]
    output = OutputClient(timeout=1)
    control = ControlPlaneClient(timeout=1)
    owner = 'tts-qnn-smoke-' + uuid.uuid4().hex[:12]
    root = args.report_dir.resolve() / owner
    root.mkdir(parents=True, exist_ok=False)
    root.chmod(0o700)
    record = {'schema': 'vq.tts-qnn-context-probe.v1',
              'component': 'pose_landmarks_control' if args.litert_control else args.component,
              'backend': 'LiteRT QNN HTP control' if args.litert_control else 'QNN HTP only', 'cpu_fallback': False,
              'synthetic_inputs_only': True, 'full_tts': False,
              'played_audio': False, 'microphone_opened': False,
              'iterations': 1 if args.litert_control else args.iterations, 'qairt': metadata['tool_versions']['qairt'],
              'contract': contract, 'status': 'failed', 'lease_released': False,
              'model_sha256': hashlib.sha256((model / component).read_bytes()).hexdigest()}
    lease = None
    container_submitted = False
    previous_handlers = {sig: signal.signal(sig, interrupted)
                         for sig in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)}
    try:
        require_idle(output)
        lease = control.acquire(owner, {'npu_compute': 100, 'cpu_compute': 10, 'ram_mb': 700},
                                'experiments.tts.qnn', priority=20, ttl_seconds=90)
        require_idle(output)
        entries = []
        for name, info in contract['inputs'].items():
            if not name.replace('_', '').isalnum():
                raise ValueError('unsafe tensor name')
            path = root / (name + '.raw')
            tensor_fixture(info, name).tofile(path)
            entries.append(name + ':=' + str(path))
        input_list = root / 'inputs.txt'
        input_list.write_text((' '.join(entries) + '\n') * args.iterations)
        environment = os.environ.copy()
        environment['LD_LIBRARY_PATH'] = str(sdk) + ':/usr/lib/aarch64-linux-gnu'
        environment['ADSP_LIBRARY_PATH'] = str(sdk) + ';/usr/lib/dsp/cdsp;/usr/lib/rfsa/adsp;/lib/rfsa/adsp;/dsp'
        if dsp_extra:
            environment['ADSP_LIBRARY_PATH'] = str(dsp_extra) + ';' + environment['ADSP_LIBRARY_PATH']
        record['dsp_library_path'] = environment['ADSP_LIBRARY_PATH']
        command = [str(sdk / 'qnn-net-run'), '--backend', str(sdk / 'libQnnHtp.so'),
                   '--retrieve_context', str(model / component), '--input_list', str(input_list),
                   '--output_dir', str(root / 'outputs'), '--use_native_input_files',
                   '--use_native_output_files', '--profiling_level', 'basic']
        record['command'] = command
        if args.container:
            # Isolated one-off process using an already installed runtime image.
            # No production container modification, network or privileged mode.
            container_inputs = root / 'container-inputs.txt'
            container_inputs.write_text(input_list.read_text().replace(str(root), '/vq-run'))
            image = 'ghcr.io/arduino/app-bricks/pose-estimation-runner:0.13.0rc1'
            command = ['docker', 'run', '--rm', '--network', 'none', '--pull', 'never',
                       '--name', owner, '--device', '/dev/fastrpc-cdsp',
                       '--device', '/dev/dma_heap/system',
                       '-v', str(sdk) + ':/vq-sdk:ro', '-v', str(model) + ':/vq-model:ro',
                       '-v', str(root) + ':/vq-run:rw',
                       '-v', '/usr/lib/dsp/cdsp:/vq-platform-dsp:ro',
                       # FastRPC's platform-shell resolver uses this fixed
                       # path; an alias mount at /vq-platform-dsp is insufficient.
                       '-v', '/usr/lib/dsp:/usr/lib/dsp:ro',
                       '-v', '/usr/share/qcom:/run/host-qcom:ro',
                       '-e', 'LD_LIBRARY_PATH=/vq-sdk:/usr/lib',
                       '-e', 'ADSP_LIBRARY_PATH=/vq-sdk;/vq-platform-dsp;/usr/lib/rfsa/adsp;/usr/lib/dsp/cdsp',
                       '--entrypoint', '/qairt-entrypoint.sh', image, '/vq-sdk/qnn-net-run',
                       '--backend', '/vq-sdk/libQnnHtp.so',
                       '--retrieve_context', '/vq-model/' + component,
                       '--input_list', '/vq-run/container-inputs.txt',
                       '--output_dir', '/vq-run/outputs', '--use_native_input_files',
                       '--use_native_output_files', '--profiling_level', 'basic']
            record['execution_environment'] = 'one-off container with existing image; no privileged mode'
            record['container_bootstrap'] = 'Arduino qairt-entrypoint.sh; host qcom read-only, merged FastRPC configuration in disposable container layer'
            if dsp_extra:
                command[command.index('--entrypoint'):command.index('--entrypoint')] = ['-v', str(dsp_extra) + ':/vq-extra-dsp:ro']
                old_path = 'ADSP_LIBRARY_PATH=/vq-sdk;/vq-platform-dsp;/usr/lib/rfsa/adsp;/usr/lib/dsp/cdsp'
                command[command.index(old_path)] = 'ADSP_LIBRARY_PATH=/vq-extra-dsp;' + old_path.split('=', 1)[1]
            command = select_container_runtime(command, args.platform_runtime)
            if args.direct_platform:
                command = direct_platform_view(command)
                record['container_bootstrap'] = 'Already prepared host qcom view mounted read-only; no writes or initialization script'
            if ioctl_trace_lib:
                index = command.index('--entrypoint')
                command[index:index] = ['-v', str(ioctl_trace_lib) + ':/vq-ioctl-trace.so:ro']
                index = command.index('/vq-sdk/qnn-net-run')
                command[index:index] = ['/usr/bin/env', 'LD_PRELOAD=/vq-ioctl-trace.so']
                record['ioctl_trace_library_sha256'] = hashlib.sha256(ioctl_trace_lib.read_bytes()).hexdigest()
            if args.unsigned_pd:
                device_config = unsigned_device_config()
                (root / 'htp-device.json').write_text(json.dumps(device_config))
                extension_config = {'backend_extensions': {
                    'shared_library_path': '/usr/lib/libQnnHtpNetRunExtensions.so',
                    'config_file_path': '/vq-run/htp-device.json'}}
                (root / 'backend-extensions.json').write_text(json.dumps(extension_config))
                command += ['--config_file', '/vq-run/backend-extensions.json']
                record['explicit_device_config'] = device_config
            if args.diagnostic:
                command = instrument_container(command)
                record['diagnostic'] = 'own child strace + loader paths + verbose QNN; 8 MiB file cap; no production attachment'
            if args.litert_control:
                control_dir = Path('/home/arduino/.local/share/vq-pose-npu-lab/candidates/mediapipe-pose-task-submodels-20230427')
                control_model = control_dir / 'pose_landmarks_detector.tflite'
                (root / 'litert-control.py').write_text(litert_control_source())
                index = command.index('--entrypoint')
                command[index:index] = ['-v', str(control_dir) + ':/vq-pose-control:ro']
                index = command.index('/vq-sdk/qnn-net-run')
                command[index:] = ['/opt/venv/bin/python3', '-B', '/vq-run/litert-control.py']
                record['component'] = 'pose_landmarks_control'
                record['backend'] = 'LiteRT QNN HTP control'
                record['contract'] = 'Existing pose model; one synthetic zero input; default CPU delegates disabled'
                record['model_sha256'] = hashlib.sha256(control_model.read_bytes()).hexdigest()
            record['runtime_source'] = ('installed vendor HTP libraries; CLI remains public QAIRT 2.45; compatibility not assumed'
                                        if args.platform_runtime else 'integrity-checked public QAIRT 2.45 libraries')
            record['container_library_environment'] = container_environment(command)
            record['command'] = command
        began = time.monotonic()
        with (root / 'native.log').open('wb') as log:
            container_submitted = args.container
            process = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, timeout=args.timeout_seconds)
        record['process_wall_ms'] = round((time.monotonic() - began) * 1000, 3)
        record['returncode'] = process.returncode
        if process.returncode != 0:
            raise RuntimeError('QNN execution failed; no automatic CPU retry')
        outputs = []
        for index in range(0 if args.litert_control else args.iterations):
            for name, info in contract['outputs'].items():
                raw = native_output_path(root, index, name)
                dtype = {'uint16': '<u2', 'float32': '<f4'}[info['dtype']]
                values = np.fromfile(raw, dtype=dtype)
                if values.size != math.prod(info['shape']):
                    raise ValueError('QNN output size mismatch')
                if info['dtype'] == 'uint16':
                    enc = info['quantization_parameters']
                    values = (values.astype(np.float32) - enc['zero_point']) * enc['scale']
                if not np.isfinite(values).all():
                    raise ValueError('nonfinite output')
                outputs.append({'iteration': index, 'name': name, 'bytes': raw.stat().st_size,
                                'min': float(values.min()), 'max': float(values.max()),
                                'finite': True, 'nonzero_values': int(np.count_nonzero(values))})
        record['outputs'] = outputs
        if args.litert_control:
            record['control_result'] = json.loads((root / 'litert-control.json').read_text())
        record['status'] = 'completed'
    except Exception as exc:
        record['error_type'] = type(exc).__name__
        record['error'] = str(exc)
    finally:
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        process_stopped = True
        if container_submitted:
            try:
                process_stopped = cleanup_owned_container(owner)
                record['owned_container_exit_confirmed'] = process_stopped
            except Exception as exc:
                process_stopped = False
                record['status'] = 'failed'
                record['cleanup_error'] = type(exc).__name__ + ': ' + str(exc)
                record['owned_container_exit_confirmed'] = False
        if lease and process_stopped:
            try:
                record['lease_released'] = bool(control.release(lease['lease_id'], owner).get('released'))
            except Exception:
                record['lease_release_unknown'] = True
            if not record['lease_released']:
                record['status'] = 'failed'
                record['cleanup_error'] = 'lease release not confirmed'
        elif lease:
            record['lease_release_deferred_to_ttl'] = True
        try:
            require_idle(output)
            record['output_idle_after'] = True
        except Exception:
            record['output_idle_after'] = False
            record['status'] = 'failed'
            record['cleanup_error'] = 'Output unavailable or not idle after probe'
        path = root / 'report.json'
        try:
            path.write_text(json.dumps(record, indent=2))
            path.chmod(0o600)
            print(json.dumps({'status': record['status'], 'component': record['component'],
                              'error': record.get('error'), 'wall_ms': record.get('process_wall_ms'),
                              'report_path': str(path)}), flush=True)
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
    return 0 if record['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--component', choices=('decoder', 'encoder', 'flow', 'bert_wrapper'), default='decoder')
    parser.add_argument('--dsp-extra-dir', type=Path, help='Optional matching, integrity-checked DSP core library directory')
    parser.add_argument('--iterations', type=int, choices=range(1, 11), default=5)
    parser.add_argument('--container', action='store_true', help='Use an isolated existing-image runtime, never a production container')
    parser.add_argument('--platform-runtime', action='store_true', help='Only with --container: test installed vendor HTP libraries, no replacements')
    parser.add_argument('--diagnostic', action='store_true', help='Only with --container: trace our child loader and FastRPC syscalls')
    parser.add_argument('--unsigned-pd', action='store_true', help='Only with --container --platform-runtime: explicit unsigned device control via installed HTP extension')
    parser.add_argument('--ioctl-trace-lib', type=Path, help='Optional locally built diagnostic-only ioctl metadata shim; no request mutation')
    parser.add_argument('--litert-control', action='store_true', help='Only with isolated vendor runtime: initialize and run existing pose model on zeros, no camera')
    parser.add_argument('--direct-platform', action='store_true', help='Only in isolated container: reuse already prepared host qcom view read-only, no bootstrap writes')
    parser.add_argument('--timeout-seconds', type=int, choices=range(5, 61), default=45,
                        help='Bound the owned native/container process; maintenance controls use 15 seconds')
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
