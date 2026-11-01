"""Bounded semantic observations with provenance, confidence and freshness."""

from __future__ import annotations

import math
import re
import threading
import time
from copy import deepcopy


KINDS = frozenset({"face_presence", "identity", "objects", "posture", "affect_cue"})
AFFECT_LABELS = frozenset({"neutral", "happy_like", "surprised_like", "sad_like",
                          "angry_like", "disgust_like", "fear_like", "contempt_like"})
IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:@-]{1,128}$")
MAX_RECORDS = 32
MAX_OBJECT_LABELS = 20
MAX_TRACKS = 16


class ContextError(ValueError):
    pass


def _number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContextError(f"invalid {name}")
    value = float(value)
    if minimum is not None and value < minimum:
        raise ContextError(f"invalid {name}")
    if maximum is not None and value > maximum:
        raise ContextError(f"invalid {name}")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ContextError(f"invalid {name}")
    return value


def _label(value, name):
    if (not isinstance(value, str) or not 1 <= len(value) <= 100
            or any(not (character.isalnum() or character in " _.-:@") for character in value)):
        raise ContextError(f"invalid {name}")
    return value


class SemanticContext:
    """Latest observation per provider/kind/source; no raw media is accepted or persisted."""

    def __init__(self, *, clock=time.time, maximum=MAX_RECORDS):
        self.clock = clock
        self.maximum = maximum
        self.lock = threading.RLock()
        self.records = {}
        self.revision = 0

    def observe(self, *, provider, kind, source, observed_at, ttl_seconds,
                confidence=None, inference=False, data=None):
        provider = _identifier(provider, "provider")
        source = _identifier(source, "source")
        if kind not in KINDS:
            raise ContextError("invalid observation kind")
        observed_at = _number(observed_at, "observed_at", minimum=0)
        ttl_seconds = _number(ttl_seconds, "ttl_seconds", minimum=0.5, maximum=120)
        if confidence is not None:
            confidence = _number(confidence, "confidence", minimum=0, maximum=1)
        if not isinstance(inference, bool) or not isinstance(data, dict):
            raise ContextError("invalid observation")
        normalized = self._normalize(kind, data)
        now = self.clock()
        if observed_at > now + 5:
            raise ContextError("observation timestamp is in the future")
        record = {
            "provider": provider, "kind": kind, "source": source,
            "observed_at": observed_at, "expires_at": observed_at + ttl_seconds,
            "confidence": confidence, "inference": inference, "data": normalized,
        }
        key = (provider, kind, source)
        with self.lock:
            previous = self.records.get(key)
            if previous is not None and previous["observed_at"] > observed_at:
                return {"accepted": False, "reason": "older_than_current", "revision": self.revision}
            self.records[key] = record
            self.revision += 1
            self._prune(now)
            if len(self.records) > self.maximum:
                oldest = min(self.records, key=lambda item: self.records[item]["observed_at"])
                del self.records[oldest]
        return {"accepted": True, "revision": self.revision}

    def snapshot(self):
        now = self.clock()
        with self.lock:
            self._prune(now)
            records = [deepcopy(value) for value in self.records.values()]
            revision = self.revision
        records.sort(key=lambda value: (value["kind"], value["source"], value["provider"]))
        people = self._people(records)
        identities = self._identities(records, people)
        objects = self._objects(records)
        posture = self._posture(records)
        affect = self._affect(records, people)
        observations = [self._metadata(value, now) for value in records]
        return {
            "revision": revision, "generated_at": now,
            "people": people, "identities": identities,
            "objects": objects, "posture": posture, "affect_cues": affect,
            "observations": observations,
            "fresh": bool(records),
            "privacy": {
                "raw_media_present": False, "embeddings_present": False,
                "persistent_history": False,
            },
        }

    def stats(self):
        snapshot = self.snapshot()
        return {
            "ready": True, "revision": snapshot["revision"],
            "active_observations": len(snapshot["observations"]),
            "kinds": sorted({item["kind"] for item in snapshot["observations"]}),
            "generated_at": snapshot["generated_at"],
        }

    def _prune(self, now):
        expired = [key for key, value in self.records.items() if value["expires_at"] <= now]
        for key in expired:
            del self.records[key]
        if expired:
            self.revision += 1

    @staticmethod
    def _metadata(record, now):
        value = {
            key: record[key] for key in (
                "provider", "kind", "source", "observed_at", "expires_at",
                "confidence", "inference",
            )
        } | {"freshness_seconds": round(max(0, now - record["observed_at"]), 3)}
        if record["kind"] == "face_presence":
            tracks = record["data"]["tracks"]
            value["confirmed_count"] = sum(t["state"] == "confirmed" for t in tracks)
            value["observed_state"] = ("observed_absent" if not tracks else
                                       "present" if value["confirmed_count"] else "unknown")
        return value

    @staticmethod
    def _normalize(kind, data):
        if kind == "face_presence":
            if set(data) != {"tracks"}:
                raise ContextError("invalid face presence fields")
            tracks = data.get("tracks", [])
            if not isinstance(tracks, list) or len(tracks) > MAX_TRACKS:
                raise ContextError("invalid face tracks")
            normalized = []
            for track in tracks:
                if not isinstance(track, dict) or track.get("state") not in {"candidate", "confirmed"}:
                    raise ContextError("invalid face track")
                normalized.append({
                    "track_id": _identifier(track.get("track_id"), "track_id"),
                    "state": track["state"],
                    "confidence": _number(track.get("confidence"), "track confidence", minimum=0, maximum=1),
                })
            return {"tracks": normalized}
        if kind == "identity":
            if set(data) - {"status", "label", "consent"}:
                raise ContextError("invalid identity fields")
            status = data.get("status")
            if status not in {"known", "unknown", "no_face", "confirming", "low_quality"}:
                raise ContextError("invalid identity status")
            value = {"status": status}
            if status == "known":
                label = data.get("label")
                if not isinstance(label, str) or not label.strip() or len(label.strip()) > 96:
                    raise ContextError("invalid identity label")
                if data.get("consent") != "enrolled_local":
                    raise ContextError("identity consent is required")
                value.update(label=label.strip(), consent="enrolled_local")
            return value
        if kind == "objects":
            if set(data) - {"counts", "confidences", "model"}:
                raise ContextError("invalid object fields")
            counts = data.get("counts", {})
            confidences = data.get("confidences", {})
            if not isinstance(counts, dict) or not isinstance(confidences, dict) or len(counts) > MAX_OBJECT_LABELS:
                raise ContextError("invalid object summary")
            values = {}
            for label, count in counts.items():
                _label(label, "object label")
                if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 100:
                    raise ContextError("invalid object count")
                confidence = confidences.get(label)
                values[label] = {
                    "count": count,
                    "confidence": None if confidence is None else _number(
                        confidence, "object confidence", minimum=0, maximum=1,
                    ),
                }
            model = data.get("model")
            if model is not None:
                _identifier(model, "model")
            return {"items": values, "model": model}
        if kind == "posture":
            if set(data) - {"state", "reasons", "enabled", "calibrated", "generation", "source_state", "frame_sequence"}:
                raise ContextError("invalid posture fields")
            state = data.get("state")
            if state not in {
                "model_loading", "observing", "unknown", "calibrating", "healthy",
                "suspect", "unhealthy", "paused", "model_unavailable",
            }:
                raise ContextError("invalid posture state")
            reasons = data.get("reasons", [])
            if not isinstance(reasons, list) or len(reasons) > 8:
                raise ContextError("invalid posture reasons")
            return {
                "state": state,
                "reasons": [_identifier(value, "posture reason") for value in reasons],
                "enabled": bool(data.get("enabled")),
                "calibrated": bool(data.get("calibrated")),
                "generation": data.get("generation"),
                "source_state": data.get("source_state", "unknown"),
                "frame_sequence": data.get("frame_sequence"),
            }
        if kind == "affect_cue":
            if set(data) != {"track_id", "top_label", "confidence", "distribution", "model"}:
                raise ContextError("invalid affect fields")
            track_id = _identifier(data["track_id"], "track_id")
            label = data["top_label"]
            distribution = data["distribution"]
            if label not in AFFECT_LABELS or not isinstance(distribution, dict) or set(distribution) != AFFECT_LABELS:
                raise ContextError("invalid affect distribution")
            scores = {key: _number(value, "affect score", minimum=0, maximum=1)
                      for key, value in distribution.items()}
            if abs(sum(scores.values()) - 1) > .02 or max(scores, key=scores.get) != label:
                raise ContextError("invalid affect probabilities")
            cue_confidence = _number(data["confidence"], "affect confidence", minimum=0, maximum=1)
            if abs(cue_confidence - scores[label]) > .02:
                raise ContextError("inconsistent affect confidence")
            return {"track_id": track_id, "top_label": label,
                    "distribution": scores, "model": _identifier(data["model"], "model")}
        raise ContextError("invalid observation kind")

    def _people(self, records):
        people = []
        face_records = [value for value in records if value["kind"] == "face_presence"]
        for record in face_records:
            for track in record["data"]["tracks"]:
                if track["state"] != "confirmed":
                    continue
                people.append({
                    "track_id": track["track_id"], "source": record["source"],
                    "confidence": track["confidence"], "identity": None,
                    "observed_at": record["observed_at"], "expires_at": record["expires_at"],
                    "provenance": {"provider": record["provider"], "inference": True},
                })
        return people

    @staticmethod
    def _identities(records, people):
        identities = []
        identity_records = [value for value in records if value["kind"] == "identity"
                            and value["data"]["status"] == "known"]
        for identity in identity_records:
            candidates = [value for value in people if value["source"] == identity["source"]]
            associated_track_id = None
            if len(candidates) == 1:
                associated_track_id = candidates[0]["track_id"]
                candidates[0]["identity"] = {
                    "label": identity["data"]["label"], "consent": "enrolled_local",
                    "confidence": identity["confidence"], "source": identity["source"],
                    "associated_track_id": associated_track_id,
                    "observed_at": identity["observed_at"], "expires_at": identity["expires_at"],
                    "provenance": {
                        "provider": identity["provider"], "inference": True,
                        "association": "single_face_same_source",
                    },
                }
            identities.append({
                "label": identity["data"]["label"], "consent": "enrolled_local",
                "confidence": identity["confidence"], "source": identity["source"],
                "associated_track_id": associated_track_id,
                "observed_at": identity["observed_at"], "expires_at": identity["expires_at"],
                "provenance": {"provider": identity["provider"], "inference": True},
            })
        return identities

    @staticmethod
    def _objects(records):
        values = []
        for record in records:
            if record["kind"] != "objects":
                continue
            for label, item in record["data"]["items"].items():
                values.append({
                    "label": label, "count": item["count"], "confidence": item["confidence"],
                    "source": record["source"], "observed_at": record["observed_at"],
                    "expires_at": record["expires_at"],
                    "provenance": {"provider": record["provider"], "inference": True,
                                   "model": record["data"].get("model")},
                })
        return sorted(values, key=lambda value: (value["source"], value["label"]))

    @staticmethod
    def _posture(records):
        values = [value for value in records if value["kind"] == "posture"]
        if not values:
            return None
        record = max(values, key=lambda value: value["observed_at"])
        return {
            **record["data"], "confidence": record["confidence"], "source": record["source"],
            "observed_at": record["observed_at"], "expires_at": record["expires_at"],
            "provenance": {"provider": record["provider"], "inference": True},
        }

    @staticmethod
    def _affect(records, people):
        cues = []
        for record in records:
            if record["kind"] != "affect_cue":
                continue
            track_id = record["data"]["track_id"]
            matches = [person for person in people if person["source"] == record["source"]
                       and person["track_id"] == track_id]
            if len(matches) != 1 or len([person for person in people
                                          if person["source"] == record["source"]]) != 1:
                continue
            cues.append({
                "track_id": track_id, "source": record["source"],
                "top_label": record["data"]["top_label"],
                "distribution": record["data"]["distribution"],
                "confidence": record["confidence"],
                "observed_at": record["observed_at"], "expires_at": record["expires_at"],
                "provenance": {"provider": record["provider"], "model": record["data"]["model"],
                               "inference": True, "association": "confirmed_single_face_track"},
            })
        return cues
