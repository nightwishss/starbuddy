"""In-process, bounded adapters for semantic fusion and posture shadow decisions.

The existing perception poller calls tick. This host has no worker, model,
hardware, output or persistence side effects.
"""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import threading
import time
import urllib.request
from collections import OrderedDict
from copy import deepcopy

from .core import Access, EventCursor
from .brief import SemanticBrief
from .policy import posture_proposal
from .engagement import InteractionArbiter


DISPLAY_STATE_URL = "http://127.0.0.1:8765/api/interaction/state"
POSTURE_STATE_URL = "http://127.0.0.1:8792/api/posture/status"
SESSION = re.compile(r"^[A-Za-z0-9_.:-]{1,192}$")
TASK_TOPICS = tuple("task." + name for name in (
    "created", "scheduled", "started", "completed", "failed", "cancelled", "cancelling"))
OUTPUT_TOPICS = tuple("audio.output." + name for name in (
    "queued", "synthesizing", "started", "completed", "failed", "cancelled", "interrupted", "expired"))
TOPICS = TASK_TOPICS + OUTPUT_TOPICS


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class FusionHost:
    def __init__(self, core, *, output=None, events=None, resources=None, conversation=None,
                 opener=urllib.request.urlopen, clock=time.time, mode=None,
                 semantic_provider=None):
        from control_plane import ControlPlaneClient
        from conversation_service import ConversationClient
        from event_service import EventClient
        from output_service import OutputClient
        self.core, self.clock, self.opener = core, clock, opener
        self.output = output if output is not None else OutputClient(timeout=.15)
        self.events = events if events is not None else EventClient(timeout=.15)
        self.resources = resources if resources is not None else ControlPlaneClient(timeout=.15)
        self.conversation = conversation if conversation is not None else ConversationClient(timeout=.15)
        self.mode = mode if mode is not None else os.environ.get("VQ_CONTEXT_POSTURE_MODE", "shadow")
        if self.mode not in {"off", "shadow", "enabled"}:
            raise ValueError("invalid posture fusion mode")
        self.semantic_provider = semantic_provider
        self.cursor = EventCursor(core)
        self.lock = threading.RLock()
        self.active_session = None
        self.last_decision = None
        self.last_semantic = {}
        self.resource_leases = []
        self.tickets = OrderedDict()
        self.closed = False
        self.interaction = InteractionArbiter(clock)
        self.wave_live = {}
        self.control_boundary = None
        from .greeting import GreetingFusion
        self.greetings = GreetingFusion(self)
        self.counters = {"ticks": 0, "invalid_observations": 0, "probe_failures": 0,
                         "event_failures": 0, "decisions": 0, "guards_issued": 0,
                         "guard_denied": 0, "last_tick_ms": 0}
        self.brief = SemanticBrief(clock=clock,
            enabled=os.environ.get("VQ_CONTEXT_BRIEF_ENABLED", "1") == "1")

    def _read_json(self, url):
        with self.opener(url, timeout=.15) as response:
            raw = response.read(32769)
        if len(raw) > 32768:
            raise ValueError("status response too large")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("invalid status response")
        return value

    def _observe(self, **values):
        try:
            return self.core.observe(**values)
        except (ValueError, TypeError, KeyError):
            self.counters["invalid_observations"] += 1
            return False

    def _semantic_record(self, item, *, kind, data, state="present", subject=None):
        observed, expires = item.get("observed_at"), item.get("expires_at")
        if not _number(observed) or not _number(expires):
            self.counters["invalid_observations"] += 1
            return
        provider = item.get("provider") or (item.get("provenance") or {}).get("provider")
        self._observe(provider=provider or "semantic-context", kind=kind,
                      source=item.get("source"), observed_at=observed,
                      ttl_seconds=expires - observed, quality=item.get("confidence"),
                      subject=subject, state=state, data=data, origin="semantic")

    def _posture(self, item):
        enabled = item.get("enabled") is True
        source_state = item.get("source_state", "unknown")
        state = "present" if source_state == "fresh" else "unknown"
        if not enabled:
            state = "disabled"
        elif source_state in {"unavailable", "disabled"}:
            state = source_state
        data = {key: item[key] for key in ("state", "reasons", "enabled", "calibrated", "generation")
                if key in item} if state == "present" else {}
        self._semantic_record(item, kind="posture", state=state, data=data)

    def _reconcile_live_posture(self, value):
        """Bridge a cached miss using only the same trusted posture producer.

        Keep its existing provider identity: a second provider would conflict
        with the cached record instead of updating the same observation lane.
        Unknown, simulation and expired samples never become permissions.
        """
        if (not isinstance(value, dict) or value.get("enabled") is not True
                or value.get("calibrated") is not True or value.get("state") != "unhealthy"
                or value.get("source_state") != "fresh"
                or value.get("simulation") is not False or value.get("simulation_result") is not False
                or type(value.get("generation")) is not int or value["generation"] < 0
                or not _number(value.get("observed_at"))
                or not 0 <= self.clock() - value["observed_at"] < 3):
            return False
        reasons = value.get("reasons")
        reasons = ([reason for reason in reasons[:20] if isinstance(reason, str)
                    and reason in {"head_forward", "torso_lean", "shoulder_asymmetry", "too_close"}]
                   if isinstance(reasons, list) else [])
        self._posture({"provider": "posture-policy", "source": "camera.environment",
                       "observed_at": value["observed_at"], "expires_at": value["observed_at"] + 3,
                       "confidence": .9, "source_state": "fresh", "state": "unhealthy",
                       "enabled": True, "calibrated": True, "generation": value["generation"],
                       "reasons": reasons})
        return True

    def _ingest_semantic(self, snapshot):
        # Presence is based on an explicit producer observation, never on a
        # missing people list or on inference about the other camera.
        people = snapshot.get("people") or []
        for item in (snapshot.get("observations") or [])[:64]:
            if not isinstance(item, dict) or item.get("kind") != "face_presence":
                continue
            state = item.get("observed_state", "unknown")
            count = item.get("confirmed_count")
            data = {}
            if state in {"present", "observed_absent"} and type(count) is int:
                data = {"count": count, "track_ids": [p["track_id"] for p in people[:32]
                        if isinstance(p, dict) and p.get("source") == item.get("source")
                        and isinstance(p.get("track_id"), str)][:20]}
            self._semantic_record(item, kind="presence", state=state, data=data)
        posture = snapshot.get("posture")
        if isinstance(posture, dict):
            self._posture(posture)
        for identity in (snapshot.get("identities") or [])[:16]:
            if not isinstance(identity, dict):
                continue
            track = identity.get("associated_track_id")
            # Identity remains source-scoped and inaccessible to default Access.
            self._semantic_record(identity, kind="identity", subject=track,
                                  data={"label": identity.get("label"), "track_id": track,
                                        "consent": identity.get("consent")})
        grouped = {}
        for item in (snapshot.get("objects") or [])[:64]:
            if not isinstance(item, dict):
                continue
            key = (item.get("source"), item.get("observed_at"), item.get("expires_at"),
                   (item.get("provenance") or {}).get("provider"))
            record = grouped.setdefault(key, {"item": item, "counts": {}})
            if isinstance(item.get("label"), str) and type(item.get("count")) is int:
                record["counts"][item["label"]] = item["count"]
        for record in list(grouped.values())[:16]:
            self._semantic_record(record["item"], kind="objects", data={"counts": record["counts"]})
        for item in (snapshot.get("affect_cues") or [])[:16]:
            if isinstance(item, dict):
                self._semantic_record(item, kind="affect", subject=item.get("track_id"),
                                      data={"label": item.get("top_label"), "track_id": item.get("track_id")})
        self._provider_availability(snapshot.get("providers") or {})

    def _provider_availability(self, providers):
        defaults = {"posture": ("posture-policy", "posture", ["camera.environment"]),
                    "face": ("face-yunet", "presence", ["camera.front", "camera.environment"]),
                    "identity": ("face-sface", "identity", ["camera.front"])}
        for name, (provider, kind, sources) in defaults.items():
            value = providers.get(name)
            if not isinstance(value, dict):
                continue
            states = value.get("sources") or [{"source": source, "ready": value.get("ready")}
                                               for source in sources]
            for source in states[:4]:
                if not isinstance(source, dict) or not source.get("source"):
                    continue
                state = ("disabled" if value.get("enabled") is False else
                         "unavailable" if value.get("ready") is False or not source.get("ready") else None)
                if state is not None:
                    self._observe(provider=provider, kind=kind, source=source["source"],
                                  observed_at=self.clock(), ttl_seconds=3, state=state,
                                  data={}, origin="semantic")

    def _poll_interaction(self):
        try:
            value = self._read_json(DISPLAY_STATE_URL)
            session = value.get("session_id")
            if (not isinstance(session, str) or not SESSION.fullmatch(session)
                    or not _number(value.get("observed_at"))
                    or value["observed_at"] > self.clock() + 1
                    or value.get("phase") not in {"idle", "listening", "thinking", "speaking", "acting", "attention", "error"}
                    or type(value.get("enabled", True)) is not bool
                    or type(value.get("output_pending", False)) is not bool
                    or type(value.get("quiet")) is not bool
                    or type(value.get("user_busy")) is not bool):
                raise ValueError("invalid interaction session")
            phase = value["phase"]
            epoch, revision = value.get("control_epoch"), value.get("control_revision")
            if (not isinstance(epoch, str) or not re.fullmatch(r"[a-f0-9]{32}", epoch)
                    or type(revision) is not int or revision < 0):
                raise ValueError("interaction control boundary unavailable")
            self.control_boundary = (epoch, revision)
            if phase == "idle" and value.get("output_pending") is True:
                phase = "acting"  # cancellation is not native cleanup acknowledgement
            self.active_session = session
            self._observe(provider="display-state", kind="interaction", source="conversation",
                          session_id=session, observed_at=value.get("observed_at"), ttl_seconds=3,
                          state="disabled" if value.get("enabled") is False else "present",
                          data={"phase": phase, "turn_id": value.get("turn_id"),
                                **{k: value[k] for k in ("quiet", "user_busy") if k in value}}, origin="poll")
        except Exception:
            self.counters["probe_failures"] += 1
            self._observe(provider="display-state", kind="interaction", source="conversation",
                          session_id=self.active_session, observed_at=self.clock(), ttl_seconds=3,
                          state="unavailable", data={}, origin="poll")

    def _poll_output(self):
        try:
            value = self.output.system_status()
            if (not isinstance(value, dict) or value.get("ready") is not True
                    or any(value.get(key, False) is not False
                           for key in ("output_quarantined", "tts_quarantined"))):
                raise ValueError("output unavailable")
            queued = value.get("queued")
            # A missing queue count/current field is unknown, not evidence of
            # idle. Both fields are provided by the existing Output API.
            if (type(queued) is not int or not 0 <= queued <= 1_000_000
                    or "current" not in value):
                raise ValueError("invalid output queue")
            current = value["current"]
            if current is not None and (not isinstance(current, dict) or not current
                    or not isinstance(current.get("job_id"), str)
                    or not 1 <= len(current["job_id"]) <= 128
                    or current.get("state") not in {"queued", "synthesizing", "playing",
                        "completed", "failed", "cancelled", "interrupted", "expired"}):
                raise ValueError("invalid current output")
            current = current or {}
            self.interaction.receipt(current.get("job_id"), current.get("state"))
            self._observe(provider="output-state", kind="output", source="output-arbiter",
                          observed_at=self.clock(), ttl_seconds=3,
                          data={"busy": bool(current) or queued > 0, "queued": queued,
                                "state": current.get("state", "queued" if queued else "idle"),
                                "job_id": current.get("job_id")}, origin="poll")
        except Exception:
            self.counters["probe_failures"] += 1
            self._observe(provider="output-state", kind="output", source="output-arbiter",
                          observed_at=self.clock(), ttl_seconds=3, state="unavailable", data={})

    def _poll_intent(self):
        if self.active_session is None:
            return
        now = self.clock()
        current = [item for item in self.core.snapshot(Access(session_id=self.active_session))["items"]
                   if item["kind"] == "intent" and item["source"] == "conversation"
                   and item["state"] == "present" and item["data"].get("snooze_until", 0) > now]
        try:
            value = self.conversation.context(self.active_session)
            if value.get("session_id") != self.active_session:
                raise ValueError("wrong conversation scope")
            memory = (value.get("memory") or {}).get("interaction.posture_snooze")
            until = now
            if isinstance(memory, dict):
                intent = memory.get("value") or {}
                expiry = memory.get("expires_in_seconds")
                if (memory.get("source") == "explicit-user" and isinstance(intent, dict)
                        and set(intent) <= {"until", "origin"} and intent.get("origin") == "user"
                        and _number(intent.get("until")) and _number(expiry) and expiry > 0
                        and now < intent["until"] <= now + 3600):
                    until = min(intent["until"], now + expiry)
            if current:
                # A direct snooze reaches this host before the display writes
                # working memory. Never erase it during that short handoff.
                return
            self._observe(provider="user-intent", kind="intent", source="conversation",
                          session_id=self.active_session, observed_at=now,
                          ttl_seconds=max(.1, until - now) if until > now else 3,
                          data={"snooze_until": until, "origin": "user"}, origin="working-memory")
        except Exception:
            self.counters["probe_failures"] += 1
            if not current:
                self._observe(provider="user-intent", kind="intent", source="conversation",
                              session_id=self.active_session, observed_at=now, ttl_seconds=3,
                              state="unavailable", data={"origin": "user"}, origin="working-memory")

    def _poll_resources(self):
        try:
            value = self.resources.status()
            available = value["available"]
            audio, cpu = available["audio_output"], available["cpu_compute"]
            if not _number(audio) or not _number(cpu) or min(audio, cpu) < 0:
                raise ValueError("invalid resource state")
            self.resource_leases = [item for item in (value.get("leases") or [])[:64]
                                   if isinstance(item, dict)]
            data = {"audio_available": audio > 0, "cpu_available": cpu}
            if "npu_compute" in available:
                npu = available["npu_compute"]
                if not _number(npu) or not 0 <= npu <= 100:
                    raise ValueError("invalid NPU budget")
                data["npu_available"] = npu
            self._observe(provider="control-plane", kind="resources", source="control-plane",
                          observed_at=self.clock(), ttl_seconds=3,
                          data=data)
        except Exception:
            self.resource_leases = []
            self.counters["probe_failures"] += 1
            self._observe(provider="control-plane", kind="resources", source="control-plane",
                          observed_at=self.clock(), ttl_seconds=3, state="unavailable", data={})

    def _accept_event(self, event, epoch):
        payload = event.get("payload")
        topic = event.get("topic")
        if not isinstance(payload, dict):
            return
        if topic in TASK_TOPICS and event.get("source") == "task-orchestrator":
            task_id = payload.get("task_id")
            session = payload.get("source_conversation_id")
            if (not isinstance(task_id, str) or not isinstance(session, str)
                    or not SESSION.fullmatch(session)):
                return
            values = dict(provider="task-events", kind="task", source="task-orchestrator",
                          subject=task_id, session_id=session,
                          data={"state": topic.removeprefix("task."), "task_id": task_id})
        elif topic in OUTPUT_TOPICS and event.get("source") == "output":
            job = payload.get("job_id")
            if not isinstance(job, str):
                return
            # Only a job bound to our ambient reservation can affect an episode.
            # The event cursor validates source epoch and delivery order.
            event_state = "playing" if topic == "audio.output.started" else topic.removeprefix("audio.output.")
            if (_number(event.get("created_at"))
                    and 0 <= self.clock() - event["created_at"] < 3):
                self.interaction.receipt(job, event_state)
            # A single job completing cannot establish that all robot output is
            # idle. This job-scoped evidence supplements the authoritative poll.
            values = dict(provider="output-events", kind="output", source="output-events", subject=job,
                          data={"busy": topic == "audio.output.started", "state": payload.get("state"),
                                "job_id": job})
        else:
            return
        self._observe(**values, observed_at=event.get("created_at"), ttl_seconds=3,
                      source_epoch=epoch, sequence=event.get("revision"), origin="event",
                      evidence_id=f"{epoch}:{event.get('revision')}")

    def _read_events(self):
        try:
            for _ in range(2):
                page = self.events.read(self.cursor.revision, topics=list(TOPICS), limit=64)
                if not self.cursor.consume(page, self._accept_event):
                    break
        except Exception:
            self.counters["event_failures"] += 1
            self.core.invalidate(origin="event", reason="bus_unavailable")

    def _prune_tickets(self):
        for token in list(self.tickets):
            if self.tickets[token]["expires_at"] <= self.clock():
                self.tickets.pop(token)
        while len(self.tickets) > 32:
            self.tickets.popitem(last=False)

    def tick(self, semantic_snapshot):
        with self.lock:
            if self.closed:
                return None
            started = time.monotonic()
            self.core.tick()
            self.last_semantic = deepcopy(semantic_snapshot or {})
            self._ingest_semantic(self.last_semantic)
            self._read_events()
            self._poll_interaction()
            self._poll_intent()
            self._poll_output()
            self._poll_resources()
            self.core.tick()
            self._prune_tickets()
            current = self.core.snapshot(Access(session_id=self.active_session))
            self.greetings.prune()
            try:
                self.wave_live = self._read_json('http://127.0.0.1:8765/api/wave/status')
            except Exception:
                self.wave_live = {}
            self.interaction.update(current, self._interaction_boundary(), self.wave_live)
            self.brief.update(current, self._brief_boundary())
            self.last_decision = posture_proposal(
                current, now=self.clock(),
                session_id=self.active_session)
            self.counters["ticks"] += 1
            self.counters["decisions"] += 1
            self.counters["last_tick_ms"] = round((time.monotonic() - started) * 1000, 3)
            return deepcopy(self.last_decision)

    def _brief_boundary(self):
        # Readable without Host's I/O lock; a changed boundary only discards
        # optional history. A fresh snapshot remains the source of truth.
        return (self.core.epoch, self.cursor.epoch, self.cursor.gaps,
                self.cursor.restarts, self.counters["event_failures"], self.active_session)

    def _interaction_boundary(self):
        return (self._brief_boundary(), self.control_boundary)

    def snapshot(self, session_id=None):
        # The core has its own short, non-I/O lock. Slow/offline adapters must
        # never hold up a conversation's snapshot behind the host polling lock.
        value = self.core.snapshot(Access(session_id=session_id))
        value["mode"] = self.mode
        # Re-evaluate expiry even if the host poller has stopped. Never publish
        # another session's intent or decision evidence references.
        value["decision"] = posture_proposal(value, now=self.clock(), session_id=session_id)
        value["ready"] = not self.closed
        value["brief"] = self.brief.read(value, self._brief_boundary())
        value["interaction_state"] = self.interaction.read()
        return value

    def explain(self, evidence_id, session_id=None):
        return self.core.explain(evidence_id, Access(session_id=session_id))

    def snooze(self, session_id, seconds, *, origin="user", echo=False):
        if (not isinstance(session_id, str) or not SESSION.fullmatch(session_id)
                or type(seconds) is not int or not 1 <= seconds <= 3600
                or origin != "user" or echo is not False):
            raise ValueError("invalid explicit user snooze")
        with self.lock:
            self._poll_interaction()
            if self.closed or session_id != self.active_session:
                raise ValueError("inactive interaction session")
            interaction = [item for item in self.core.snapshot(Access(session_id=session_id))["items"]
                           if item["kind"] == "interaction" and item["source"] == "conversation"]
            if not interaction or max(interaction, key=lambda item: item["arrived_at"])["state"] != "present":
                raise ValueError("interaction unavailable")
            now = self.clock()
            self.core.observe(provider="user-intent", kind="intent", source="conversation",
                              observed_at=now, ttl_seconds=seconds, session_id=session_id,
                              data={"snooze_until": now + seconds, "origin": "user"}, origin="user")
            return {"accepted": True, "session_id": session_id, "snooze_until": now + seconds,
                    "mode": self.mode, "effective": self.mode == "enabled"}

    def _refresh(self):
        semantic = self.semantic_provider() if self.semantic_provider is not None else self.last_semantic
        return self.tick(semantic)

    @staticmethod
    def _subject_signature(snapshot):
        items = [item for item in snapshot["items"] if item["kind"] == "presence"
                 and item["source"] == "camera.environment"]
        if not items:
            return None
        item = max(items, key=lambda v: v["arrived_at"])
        if item["state"] != "present" or item["data"].get("count") != 1:
            return None
        return tuple(sorted(item["data"].get("track_ids") or [])) or None

    def issue_guard(self):
        with self.lock:
            try:
                self._refresh()
            except Exception:
                return {"mode": self.mode, "decision": {"action": "defer", "reason": "fusion_unavailable"}}
            snapshot = self.snapshot(self.active_session)
            decision = snapshot["decision"]
            if (not self.closed and self.mode == "enabled" and decision.get("reason") in {
                    "no_confirmed_posture_reminder", "posture_unknown", "posture_stale"}):
                try:
                    if self._reconcile_live_posture(self._read_json(POSTURE_STATE_URL)):
                        snapshot = self.snapshot(self.active_session)
                        decision = snapshot["decision"]
                except Exception:
                    pass  # A failed bounded read cannot grant a posture ticket.
            result = {"mode": self.mode, "decision": decision}
            if self.closed or self.mode != "enabled" or decision["action"] != "propose":
                return result
            if type(decision.get("posture_generation")) is not int:
                result["decision"] = {**decision, "action": "defer", "reason": "posture_generation_unknown"}
                return result
            token = secrets.token_urlsafe(24)
            admission = self.interaction.reserve("posture", token, self.clock() + 15, snapshot)
            if not admission["allowed"]:
                result["decision"] = {**decision, "action": "defer", "reason": admission["reason"]}
                return result
            self.tickets[token] = {
                "epoch": self.core.epoch, "session_id": self.active_session,
                "subject": decision["subject"], "subject_signature": self._subject_signature(snapshot),
                "generation": decision["posture_generation"], "expires_at": self.clock() + 15,
                "job_id": None,
            }
            self._prune_tickets()
            self.counters["guards_issued"] += 1
            result["guard"] = {"kind": "posture", "token": token}
            return result

    def authorize_guard(self, guard, job_id=None):
        if (isinstance(guard, dict) and set(guard) == {"kind", "token"}
                and guard.get("kind") == "greeting"):
            return self.greetings.authorize(guard, job_id)
        with self.lock:
            def denied(reason):
                self.counters["guard_denied"] += 1
                if isinstance(guard, dict) and isinstance(guard.get("token"), str):
                    self.interaction.revoke(guard["token"])
                return {"allowed": False, "reason": reason}
            if self.closed or self.mode != "enabled":
                return denied("fusion_not_enabled")
            if (not isinstance(guard, dict) or set(guard) != {"kind", "token"}
                    or guard.get("kind") != "posture" or not isinstance(guard.get("token"), str)):
                return denied("invalid_guard")
            ticket = self.tickets.get(guard["token"])
            if ticket is None or ticket["epoch"] != self.core.epoch or ticket["expires_at"] <= self.clock():
                return denied("expired_guard")
            if job_id is not None and (not isinstance(job_id, str) or not 1 <= len(job_id) <= 128):
                return denied("invalid_job")
            if ticket["job_id"] is not None and ticket["job_id"] != job_id:
                return denied("guard_job_mismatch")
            if not self.interaction.claim(guard["token"], job_id):
                return denied("reservation_revoked")
            try:
                self._refresh()
                # Recheck the live producer to close the pause/generation race
                # between the periodic semantic poll and Output dequeue/play.
                posture = self._read_json(POSTURE_STATE_URL)
                if (posture.get("generation") != ticket["generation"]
                        or not self._reconcile_live_posture(posture)):
                    return denied("posture_changed")
            except Exception:
                return denied("fusion_unavailable")
            if ticket["session_id"] != self.active_session:
                return denied("session_changed")
            if ticket["expires_at"] <= self.clock():
                return denied("expired_guard")
            snapshot = self.core.snapshot(Access(session_id=self.active_session))
            admission = self.interaction.authorize(guard["token"], job_id, snapshot)
            if not admission["allowed"]:
                return denied(admission["reason"])
            if self._subject_signature(snapshot) != ticket["subject_signature"]:
                return denied("subject_changed")
            # Ignore only this ticket's already-dequeued Output job. The real
            # snapshot remains busy, and other jobs still block this proposal.
            own_output = False
            for item in snapshot["items"]:
                if (item["kind"] == "output" and item["source"] == "output-arbiter"
                        and item["state"] == "present" and item["data"].get("queued") == 0
                        and item["data"].get("state") in {"synthesizing", "playing"}
                        and job_id is not None and item["data"].get("job_id") == job_id):
                    item["data"]["busy"] = False
                    own_output = True
            if own_output and any(
                    lease.get("owner") == "session:posture-check"
                    and lease.get("purpose") == "posture-reminder"
                    and _number((lease.get("resources") or {}).get("audio_output"))
                    and lease["resources"]["audio_output"] > 0
                    for lease in self.resource_leases):
                for item in snapshot["items"]:
                    if item["kind"] == "resources" and item["source"] == "control-plane":
                        item["data"]["audio_available"] = True
            decision = posture_proposal(snapshot, now=self.clock(), session_id=self.active_session)
            if (decision["action"] != "propose" or decision.get("subject") != ticket["subject"]
                    or decision.get("posture_generation") != ticket["generation"]):
                return denied(decision["reason"])
            if job_id is not None:
                ticket["job_id"] = job_id
            return {"allowed": True, "reason": "revalidated", "context_version": snapshot["version"]}

    def stats(self):
        # Advisory metrics only; no network I/O or blocking host lock.
        return {**self.counters, "ready": not self.closed, "mode": self.mode,
                "tickets": len(self.tickets), "event_gaps": self.cursor.gaps,
                "event_restarts": self.cursor.restarts, "event_revision": self.cursor.revision,
                "brief": self.brief.stats(), "interaction_state": self.interaction.read()}

    def close(self):
        with self.lock:
            self.closed = True
            self.tickets.clear()
            self.interaction.close()
            self.brief.close()
