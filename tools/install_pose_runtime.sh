#!/bin/sh
set -eu

POSE_ROOT="${VQ_POSE_ROOT:-$HOME/.local/share/vq-pose-lab}"
VENV="$POSE_ROOT/venv"
MODEL="$POSE_ROOT/pose_landmarker_lite.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_SHA256="59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"

mkdir -p "$POSE_ROOT"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install \
  "mediapipe==1.0.1" "opencv-contrib-python==5.0.0.93" "pillow==12.3.0" "requests==2.34.2"

temporary="$(mktemp "$POSE_ROOT/.pose-landmarker.XXXXXX")"
trap 'rm -f "$temporary"' EXIT HUP INT TERM
curl -fL --retry 3 -o "$temporary" "$MODEL_URL"
printf '%s  %s\n' "$MODEL_SHA256" "$temporary" | sha256sum -c -
mv "$temporary" "$MODEL"
trap - EXIT HUP INT TERM
chmod 0644 "$MODEL"

echo "Pose runtime installed at $POSE_ROOT"

