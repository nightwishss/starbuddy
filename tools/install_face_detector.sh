#!/usr/bin/env bash
set -euo pipefail

model_dir="${XDG_DATA_HOME:-$HOME/.local/share}/vq-ai/models/face/yunet"
model_path="$model_dir/face_detection_yunet_2026may.onnx"
expected_sha="ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
url="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx"
temporary="$(mktemp)"
trap 'rm -f "$temporary"' EXIT

mkdir -p "$model_dir"
chmod 700 "$model_dir"
curl --fail --location --silent --show-error --output "$temporary" "$url"
printf '%s  %s\n' "$expected_sha" "$temporary" | sha256sum --check --status
install -m 600 "$temporary" "$model_path"
printf '%s\n' "$model_path"
