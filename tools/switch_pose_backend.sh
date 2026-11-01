#!/bin/sh
set -eu

usage() {
    echo "usage: $0 [--execute] cpu|npu" >&2
    exit 2
}

execute=false
if [ "${1:-}" = "--execute" ]; then
    execute=true
    shift
fi
backend=${1:-}
[ "$#" -eq 1 ] || usage
[ "$backend" = cpu ] || [ "$backend" = npu ] || usage

if [ "$backend" = npu ]; then
    actions='stop vq-pose-hybrid.service
stop vq-pose-runner.service
start vq-pose-npu-worker.service
start vq-pose-npu.service
verify http://127.0.0.1:8791/healthz backend=qnn'
else
    actions='stop vq-pose-npu.service
stop vq-pose-hybrid.service
stop vq-pose-npu-worker.service
start vq-pose-runner.service
verify http://127.0.0.1:8791/healthz backend=cpu-xnnpack'
fi

if [ "$execute" != true ]; then
    printf '%s\n' "DRY RUN: requested backend=$backend" "$actions"
    echo "Re-run with --execute only after the N4 acceptance review."
    exit 0
fi

wait_ready() {
    expected=$1
    count=0
    while [ "$count" -lt 40 ]; do
        status=$(curl -fsS --max-time 2 http://127.0.0.1:8791/healthz 2>/dev/null || true)
        case "$status" in
            *'"ready": true'*"$expected"*) return 0 ;;
        esac
        count=$((count + 1))
        sleep 1
    done
    return 1
}

wait_worker_ready() {
    count=0
    while [ "$count" -lt 40 ]; do
        status=$(curl -fsS --max-time 2 http://127.0.0.1:8793/healthz 2>/dev/null || true)
        case "$status" in
            *'"ready": true'*'"npu_accelerated": true'*) return 0 ;;
        esac
        count=$((count + 1))
        sleep 1
    done
    return 1
}

if [ "$backend" = npu ]; then
    systemctl --user stop vq-pose-hybrid.service vq-pose-runner.service
    if systemctl --user start vq-pose-npu-worker.service \
            && wait_worker_ready \
            && systemctl --user start vq-pose-npu.service \
            && wait_ready 'qnn-htp-landmark'; then
        echo "pose backend switched to NPU hybrid"
        exit 0
    fi
    echo "NPU health check failed; restoring CPU backend" >&2
    systemctl --user stop vq-pose-npu.service vq-pose-npu-worker.service || true
    systemctl --user start vq-pose-runner.service
    wait_ready 'cpu-xnnpack' || {
        echo "CRITICAL: CPU rollback health check failed" >&2
        exit 1
    }
    exit 1
fi

systemctl --user stop vq-pose-npu.service vq-pose-hybrid.service vq-pose-npu-worker.service
systemctl --user start vq-pose-runner.service
wait_ready 'cpu-xnnpack' || {
    echo "CPU backend health check failed" >&2
    exit 1
}
echo "pose backend switched to CPU"
