# Architecture

StarBuddy combines independently deployed models and deterministic host services. It is not a single end-to-end multimodal model. The host manages timing, context freshness, resource admission, and action authority so that model output can be used without treating it as an executable instruction.

## Spoken turn

An explicitly enabled conversation session follows this flow:

1. The browser requests a turn with the current control revision; the host admits one owner-bound input.
2. Runtime captures a bounded temporary input and local Whisper ASR produces text.
3. Dialogue routing uses the current utterance and bounded recent conversation. The demonstration configuration prefers the authorized cloud text API, with an installed local text-model fallback.
4. The model produces natural response text, or requests the one supported read-only visual-identity context capability.
5. The host validates the proposal, checks cancellation/currentness, and sends permitted speech through local TTS and Output.
6. The session waits for real output completion and confirmed temporary-input cleanup before requesting another input. Unconfirmed delivery, errors, stopping, or navigation prevent automatic continuation.

The browser does not create an always-on background microphone. No new capture should overlap the companion's speech. Native inference/capture may need time to finish after cancellation; the UI distinguishes stopping from confirmed idle rather than claiming immediate physical termination.

Relevant code: `gateway/display/server.py`, `gateway/display/situated.py`, `gateway/display/static/continuous-voice.js`, `gateway/llm/`, `gateway/speech/`, and `output_service/`.

## Multimodal context and Fusion

Camera-derived people, objects, gestures, and posture are observations, not commands. Source adapters timestamp them; context services expose availability and freshness; the shared workspace tracks anonymous actors, attention, goals, and recent events.

Fusion connects these inputs to interaction state and behavior arbitration. It distinguishes the active conversation from background observations and lower-priority reminders. A wave is an intentional interaction cue and may request a greeting. Ordinary people/object observations update context without continuously narrating every detection.

Missing, stale, low-confidence, or conflicting evidence stays unknown. An unrelated camera change must not invalidate a context-independent conversation such as an arithmetic or general-knowledge question. Speech, touch, and a face in the image are not automatically bound to the same person.

Relevant code: `interaction_context/`, `interaction_policy/`, `perception_context_service/`, `gateway/agent/situated.py`, and `gateway/display/situated.py`.

## Local visible identity

The dialogue model can return the exact read-only request `{"context_request":"visible_identity"}` instead of guessing a name. The local host then checks current visual evidence, enrollment/consent, source freshness, and an unambiguous visible track before composing an identity-grounded reply.

The cloud dialogue request does not contain the local face label, enrollment material, visual observations, or a personal profile. Locally generated identity replies are not folded back into ordinary cloud history. Recognition of a visible enrolled participant still does not confirm who spoke into the microphone.

Relevant code: `gateway/llm/dialogue_intent.py`, `gateway/display/face_dialogue.py`, `face_identity_service/`, and `face_vault_service/`.

## Posture guardian

The guardian compares reliable pose measurements with a confirmed personal baseline and a temporal policy. Default policy values require approximately 20 seconds of sustained deviation and enforce a 180-second reminder cooldown; uncertain visibility is not classified as bad posture.

Manual enablement is independent of the current browser page. A failed new calibration should not replace a previously confirmed baseline. Some optional cues require additional calibration; an unavailable optional cue is not proof that every baseline-supported posture feature is disabled.

A reminder is a short authored notification, not an independent background LLM conversation. It passes through interaction/resource admission and Output ownership, defers while higher-priority interaction is active, and does not claim delivery without a matching acknowledgement.

Relevant code: `perception/posture.py`, `perception/posture_calibration.py`, `posture_service/`, and the display host's `PostureController`.

## Execution and privacy boundaries

- Model proposals are untrusted. Runtime schemas, control-plane authorization, current-turn checks, and explicit task confirmation determine what can execute.
- Temporary media uses scoped handles and owners rather than granting a model arbitrary filesystem access. Output receipts distinguish accepted, pending, completed, cancelled, and unknown delivery.
- Conversation context is bounded; personal memory and face enrollment are separate local services, not a requirement to upload a reusable recording library.
- `VQ_STARBUDDY_OFFLINE_ONLY` blocks cloud use. Enabling cloud text dialogue does not enable camera, microphone, or profile export.
- Hardware-specific Router/MCU integration exists separately from the conversational host. Mechanical actions are outside the accepted demonstration scope.

This separation improves inspectability, but it is not evidence of comprehensive security, general intelligence, clinical effectiveness, or uninterrupted system reliability. See [Demo status](DEMO_STATUS.md).
