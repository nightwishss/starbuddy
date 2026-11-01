# StarBuddy

StarBuddy is a prototype desktop companion combining spoken interaction, local visual context, posture reminders, and guided activities. This repository contains the application and service source used by the demonstration system, plus tests and deployment templates, for technical review.

It is **not a complete device image or a one-command installation**. Model weights, inference engines, platform-specific drivers, credentials, personal enrollment data, and prerecorded demonstration media are not included. See [Setup](docs/SETUP.md) before attempting to run services.

## Current product shape

- An explicit start/stop button enables bounded continuous conversation: local microphone capture and ASR, text-model reasoning, then local TTS and playback. Listening does not continue over the companion's own speech.
- The current demonstration configuration uses a cloud text LLM for natural dialogue, with an installed local text-model fallback. Raw microphone audio and camera frames are not sent through this cloud dialogue route.
- Local face and gesture services provide visual context. A wave can trigger a spoken greeting; an enrolled visible participant can be recognized through the local identity boundary. A visible person is not automatically treated as the microphone speaker.
- A posture guardian tracks reliable deviations against a personal baseline and schedules gentle, cooldown-limited spoken reminders.
- Browser applications include conversation, collaborative storytelling, and expression-learning activities, with a character-led interface intended for a landscape touchscreen.

These are prototype capabilities, not claims of clinical benefit or general recognition accuracy. [Demo status](docs/DEMO_STATUS.md) distinguishes observed demonstrations from unverified scope.

## Source layout

| Area | Main directories |
| --- | --- |
| Browser UI and interaction host | `gateway/display/` |
| Text-model clients and dialogue routing | `gateway/llm/`, `gateway/agent/` |
| Shared context, state, and arbitration | `interaction_context/`, `interaction_policy/`, `orchestration/` |
| Capability execution and service SDKs | `runtime/`, `capability/`, `sdk/`, `control_plane/` |
| Audio, temporary media, and output ownership | `audio_service/`, `media_service/`, `output_service/`, `gateway/speech/` |
| Camera, face, pose, and posture adapters | `camera_stream_service/`, `face_service/`, `face_identity_service/`, `pose_service/`, `posture_service/`, `perception/` |
| Privacy, personal state, and tasks | `governance/`, `face_vault_service/`, `personal_memory_service/`, `task_orchestrator/` |
| Offline tests and deployment examples | `tests/`, `deploy/systemd/`, `tools/` |

See [Architecture](docs/ARCHITECTURE.md) for boundaries and data flow.

## Review without a board

Use Python 3.10 or newer. From the repository root, this syntax-only check reads Python source without importing services or starting models/devices:

```sh
python3 -B - <<'PY'
import ast
from pathlib import Path

paths = sorted(Path('.').rglob('*.py'))
for path in paths:
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
print(f'Parsed {len(paths)} Python files')
PY
```

The following selected tests use synthetic inputs and do not require a board, camera, microphone, speaker, or model:

```sh
PYTHONPATH=.:tests python3 -B -m unittest \
  test_posture_policy test_posture_calibration test_child_safety \
  test_conversation_store test_multimodal_workspace test_model_router
```

This is a selected regression set, not full hardware acceptance. Some other tests and bring-up tools can exercise real services or devices; inspect them before running broad discovery.

## Running an already provisioned system

After installing the external dependencies and starting the required backend services, the display host entry point is:

```sh
python3 -m gateway.display.server --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/` on the device, or use a loopback SSH tunnel from another computer:

```sh
ssh -N -L 8765:127.0.0.1:8765 <board-user>@<board-address>
```

Then open the same URL on that computer. The browser UI can also be used on a phone with an equivalent secure, loopback-preserving connection. The display control APIs enforce local/same-origin restrictions; exposing the service on a LAN address is not a supported substitute.

Detailed provisioning boundaries, optional cloud configuration, and service templates are in [Setup](docs/SETUP.md).

## Included and excluded assets

Original application character illustrations are included. There are no personal recordings, real-person enrollment images, model weights, vendor container images, or presentation slides in this source distribution.

The prerecorded story sample and its poster are omitted. The code retains the sample route; skip that sample or provide your own compatible files as described in [Setup](docs/SETUP.md#optional-story-sample). This omission does not supply or replace story-generation model dependencies.

## Usage and licensing

This repository is public source for review, not an additional blanket open-source license grant. Read [NOTICE.md](NOTICE.md), and review the terms of any separately obtained model, library, driver, or runtime before using or redistributing it.
