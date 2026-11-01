# Demonstration status

This describes prototype scope at the public-source submission snapshot (November 2026). It separates demonstrated behavior from implementation presence and from formal acceptance. A synthetic test is not a physical-device test.

## Demonstrated or participant-confirmed

- A real wave interaction triggered a spoken greeting, including a locally enrolled participant's identity where valid visual evidence was available.
- A posture deviation triggered an audible, gentle reminder in a participant-observed device demonstration.
- Natural continuous conversation is implemented with local ASR/TTS and an explicitly configured cloud text dialogue route. The session has start/stop controls and waits for output completion before continuing input.

These are bounded demonstrations, not recognition-accuracy studies or guarantees for every room, participant, lighting condition, accent, microphone, or display.

## Implemented, with external dependencies

| Area | Qualification |
| --- | --- |
| Browser conversation and activities | Source and UI are included; successful operation depends on running backend services and inference assets. |
| Story generation and speech/replay | Service/UI paths are included; text generation and local TTS must be provisioned. The optional prerecorded sample is not distributed. |
| Local face/gesture/context | Adapters and host integration are included; models, consented enrollment, and camera setup are external. |
| Posture guardian | Baseline, temporal policy, state persistence, and Output arbitration are implemented; target-device pose quality must be checked. |
| FER / expression cues | Adapter code is present; external model/runtime and broader accuracy validation are required. A cue must not be treated as a definitive emotional state. |
| Local model fallback | Integration exists; the model service, compatible assets, available memory, and latency must be validated independently. |
| System services / containers | Example units include target-specific assumptions. This is not an installable cross-board appliance or a complete Docker release. |

## Not accepted or not claimed

- Comprehensive face, gesture, object, pose, or expression accuracy across users and environments.
- Clinical effectiveness, diagnostic use, or therapeutic benefit for autistic children or any other group.
- A fully autonomous, general-purpose multimodal agent.
- An uninterrupted 72-hour stability pass, continuous inference stress test, or full-system admission certification. Earlier observation periods and unobserved gaps must not be combined into such a claim.
- MCU/RTOS mechanical-control acceptance or physical actuator safety.
- NVIDIA Jetson compatibility or equivalent performance on other hardware.
- Full offline functionality without separately provisioning and testing all required local models.

## Recommended reviewer checks

First run the syntax-only and selected synthetic tests in [README](../README.md#review-without-a-board). For a provisioned device, separately record non-private functional checks for microphone input, speaker audibility, start/stop, next-turn continuity, story speech, context freshness, and posture reminder deferral/cooldown. Record failures and unknown states as such.

Use a supervised, consent-based demonstration. Keep a stop control available; do not require eye contact, a particular facial expression, or continued participation. Do not publish enrollment material, personal audio/video, profiles, or transcripts as test evidence.
