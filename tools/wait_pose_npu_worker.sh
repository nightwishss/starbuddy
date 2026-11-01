#!/bin/sh
set -eu

count=0
while [ "$count" -lt 40 ]; do
    status=$(curl -fsS --max-time 2 http://127.0.0.1:8793/healthz 2>/dev/null || true)
    case "$status" in
        *'"ready": true'*'"npu_accelerated": true'*) exit 0 ;;
    esac
    count=$((count + 1))
    sleep 1
done
echo "QNN pose worker did not become ready within 40 seconds" >&2
exit 1
