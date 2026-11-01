# Setup and running boundaries

## 1. What this checkout provides

This is source code, tests, browser assets, and deployment examples. It does not provision a complete operating system, install inference runtimes, download models, enroll a person, or configure device permissions. There is no complete dependency lockfile or portable all-in-one Docker image in this distribution.

The demonstration target is a Linux VENTUNO Q system. Qualcomm-specific acceleration and container examples are platform-specific. NVIDIA Jetson and other boards require their own inference backend, drivers, model conversion, and resource validation; compatibility is not established by this repository.

Use Python 3.10 or newer. Python and system dependencies vary by service: examples in the source include `msgpack` for Router RPC, OpenCV/NumPy for vision, MediaPipe for the CPU pose adapter, and PyGObject with GTK 3/WebKit2 4.1 for the optional HDMI kiosk shell. Speech additionally requires native executables and separate model/runtime environments. This is not an exhaustive installation list.

Do the syntax check and selected offline tests in [README](../README.md#review-without-a-board) before touching hardware. Do not run every bring-up script as an installation step.

## 2. External model and runtime dependencies

Provision the required components separately, under their own licenses, for the services you intend to use.

| Function | Existing source integration | External requirements |
| --- | --- | --- |
| Local ASR | `gateway/speech/engine.py`, `capability/speech/impl.py` | `whisper-cli`, a compatible Whisper model, `ffmpeg` |
| Local TTS | `gateway/speech/`, `capability/speech/impl.py` | One configured Piper, Matcha, Melo, or hardware-specific Melo-QNN backend and its assets |
| Local text fallback | `gateway/llm/client.py` | Compatible local HTTP model service; defaults to loopback port 9001 and model `qwen3_4b_instruct_2507` |
| Face detection / identity | `face_service/`, `face_identity_service/`, `face_vault_service/` | Detector and recognizer models, local vault setup, explicit enrollment |
| Pose | `pose_service/` | MediaPipe model/runtime, or the hybrid adapter's separately installed acceleration worker |
| Expression cues | `affect_cues_service/` | Compatible FER model/runtime; not a substitute for the child's self-report |
| Camera / audio / display | Runtime capabilities and service adapters | Working Linux device drivers, permissions, capture/playback tools, and physical connections |

`capability/speech/impl.py` currently expects these default locations relative to the service user's home:

```text
.local/share/vq-ai/src/whisper.cpp/build/bin/whisper-cli
.local/share/vq-ai/models/stt/ggml-base.bin
.local/share/vq-ai/venv/bin/python
.local/share/vq-ai/models/tts/zh_CN-huayan-medium.onnx
.local/share/vq-tts-lab/venv/bin/python
.local/share/vq-ai/models/tts/vits-melo-tts-zh_en/
```

The Melo directory requires `model.onnx`, `lexicon.txt`, and `tokens.txt`. Other TTS backends have different assets; inspect the same source file before selecting them. These paths describe integration expectations, not bundled files or automatic downloads.

`VQ_TTS_BACKEND` selects `piper`, `matcha`, `melo`, or `melo-qnn`. The code default is `piper`; an existing Runtime unit example selects `melo`. For example, set `VQ_TTS_BACKEND=melo` in the Runtime service environment only after provisioning that backend. A setting alone does not install bilingual synthesis.

## 3. Cloud text dialogue (optional)

Cloud use requires explicit configuration and an authorized credential. Keep credentials outside the checkout, outside public artifacts, and out of shell history. The client accepts an existing, process-owner-only key file (no group/other permissions; normally mode `0600`). Do not commit that file or a populated environment file.

Configure the display service's environment, for example:

```sh
export VQ_CLOUD_LLM_ENABLED=1
export VQ_DIALOGUE_CLOUD_FIRST=1
export VQ_CLOUD_LLM_API_KEY_FILE=/absolute/path/outside/checkout/cloud-key
export VQ_CLOUD_LLM_TIMEOUT=6
```

These are environment variable examples, not credentials. The current client restricts the endpoint to `https://llm.iiia.es/v1/chat/completions`; `VQ_CLOUD_LLM_URL` is not an arbitrary-provider escape hatch. `VQ_CLOUD_LLM_MODEL` selects a compatible authorized model at that endpoint. Review `gateway/llm/cloud.py` for the current default and bounds.

The natural-dialogue route sends the current user text and bounded recent text dialogue, not raw audio, camera frames, face enrollment data, or local visual identity labels. Do not manually paste private media, profiles, or enrollment data into a cloud prompt. The host can satisfy a model-requested visible-identity lookup locally without uploading its result as cloud history.

For an offline configuration, set:

```sh
export VQ_STARBUDDY_OFFLINE_ONLY=1
```

This blocks cloud use; it does not install the local model or guarantee successful local inference. Cloud construction is disabled unless `VQ_CLOUD_LLM_ENABLED=1`. The display's cloud-first preference takes effect only when a usable cloud client is enabled and offline-only mode is not active.

## 4. Service entry points and templates

Run services as the same intended desktop user where they share user-scoped Unix sockets and temporary media. Inspect socket paths, state directories, hardware permissions, and dependencies first. Do not run the whole stack as root to bypass permission errors.

Core entry points present in the source include:

```sh
python3 -m control_plane.service
python3 -m event_service.service
python3 -m media_service.service
python3 -m runtime.service
python3 -m conversation_service.service
python3 -m output_service.service
python3 -m perception_context_service.service --poll-interval 1
python3 -m gateway.display.server --host 127.0.0.1 --port 8765
```

This list identifies modules, not a complete ordered startup script. Individual services and enabled features have additional dependencies. Runtime startup is not equivalent to a working microphone, speaker, model, or display.

Vision entry points include the following; model paths are placeholders:

```sh
python3 -m camera_stream_service.service --host 127.0.0.1 --port 8781
python3 -m face_service.service --host 127.0.0.1 --port 8794 --model /path/to/detector.onnx
python3 -m face_vault_service.service
python3 -m face_identity_service.service --host 127.0.0.1 --port 8795 \
  --detector /path/to/detector.onnx --recognizer /path/to/recognizer.onnx
python3 -m pose_service.service --host 127.0.0.1 --port 8791 \
  --model /path/to/pose.task \
  --camera-url 'http://127.0.0.1:8781/live.mjpg?consumer=pose&fps=10'
python3 -m posture_service.service --host 127.0.0.1 --port 8792
```

These commands start real services and can access devices; they are not offline review commands. Verify camera rotation and device permissions for your installation. Do not run the CPU pose service and an acceleration-backed pose service on the same port. The hybrid adapter also needs an external worker; it is not included merely because `pose_service/hybrid_service.py` is present.

`deploy/systemd/` contains reference units, not a complete auto-installer. Before installing any unit:

1. Adapt its user, working directory, Python environment, model paths, and socket/device permissions.
2. Check every `After=`, `Wants=`, and `Requires=` dependency; not all referenced units or external workers are supplied here.
3. Review hardware-specific container/device mappings independently. Do not treat the local-LLM example as a redistributable or portable vendor image.
4. For `vq-display-kiosk.service`, set `VQ_KIOSK_URI=http://127.0.0.1:8765/` when adapting the template; the supplied template contains a different service port. The Python kiosk's own default is 8765.
5. Validate one service at a time before enabling automatic startup. A process being alive is not proof of working audio, touch, HDMI, or inference.

The optional HDMI shell entry point is `python3 -m gateway.display.kiosk`. It needs the GTK/WebKit dependencies and a working graphical session. A normal browser can use the same display application without this shell.

## 5. Browser connection and controls

Use `http://127.0.0.1:8765/` locally. Applications include `/interaction`, `/posture`, `/story`, and `/masks`.

For remote development, forward the same loopback port:

```sh
ssh -N -L 8765:127.0.0.1:8765 <board-user>@<board-address>
```

Open `http://127.0.0.1:8765/` on the SSH client machine. Phone browsers require an equivalent secure connection that preserves the allowed local origin. Do not change the display binding to `0.0.0.0`, disable origin checks, or publicly expose the control APIs as a setup shortcut.

Conversation capture requires the explicit start control. Stop before changing applications, reconnecting peripherals, or diagnosing audio. Playback completion reported by software still needs a listener to verify that a speaker is physically audible.

## Optional story sample

All prerecorded demonstration video/audio media and the story sample poster are intentionally excluded. The story UI still references these optional sample assets under `gateway/display/static/story-media/`:

```text
story-garden-demo-v1.mp4
story-garden-demo-v1.webm
story-garden-demo-v1.jpg
```

Skip the prerecorded sample, or provide your own browser-compatible, appropriately licensed sample files at those names and check `gateway/display/story_media.py` for serving constraints. Do not add recordings or personal media to a public repository. The sample's absence is separate from story generation, which still needs configured text and speech services.

## Validation boundaries

Offline tests establish software behavior with synthetic inputs. Validate actual ASR, response latency, output audibility, camera orientation, touch, and resource contention on the target hardware using non-private fixtures and informed consent. Do not infer a physical or clinical pass from mocks, status endpoints, or process liveness.
