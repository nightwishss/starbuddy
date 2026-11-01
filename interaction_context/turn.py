"""Fuse existing semantic observations without running models or retaining media."""

from __future__ import annotations

import time

from .brief import sanitize_change


def _recent_changes(fusion, now):
    """Host has checked track continuity; recheck authority/time at consumption."""
    if not isinstance(fusion, dict) or fusion.get("ready") is not True:
        return []
    brief = fusion.get("brief")
    if (not isinstance(brief, dict) or brief.get("schema_version") != 1
            or brief.get("kind") != "historical_observations"
            or brief.get("basis_epoch") != fusion.get("epoch")
            or not isinstance(brief.get("prepared_at"), (int, float))
            or not brief["prepared_at"] <= now):
        return []
    rows, items = brief.get("changes"), fusion.get("items")
    if not isinstance(rows, list) or not isinstance(items, list):
        return []
    result = []
    for row in rows[-6:]:
        row = sanitize_change(row)
        if row is None or not row["observed_at"] <= now < row["retain_until"]:
            continue
        current = [item for item in items[:128] if isinstance(item, dict)
                   and item.get("source") == row["source"] and item.get("kind") == row["kind"]
                   and item.get("session_id") is None and item.get("subject") is None]
        if (len(current) != 1 or current[0].get("state") not in {"present", "observed_absent"}
                or current[0].get("conflict") or not _fresh(current[0], now)):
            continue
        # Historic source_expires_at stays historic. Separate short-lived read
        # authority prevents delayed prompt construction from reviving history.
        result.append({**row, "read_valid_until": min(row["retain_until"], current[0]["expires_at"])})
    return result


def _fresh(item, now):
    return (isinstance(item, dict)
            and isinstance(item.get("observed_at"), (int, float))
            and isinstance(item.get("expires_at"), (int, float))
            and item["observed_at"] <= now < item["expires_at"])


def _evidence(item, state="available"):
    if item is None:
        return {"state": "missing"}
    return {"state": state, "source": item.get("source"),
            "provider": (item.get("provenance") or {}).get("provider"),
            "confidence": item.get("confidence"),
            "observed_at": item.get("observed_at"),
            "expires_at": item.get("expires_at")}


def _select(item, *keys):
    return {key: item[key] for key in keys if key in item}


def assemble_turn_context(request, perception=None, identity=None, tasks=None, *, now=None):
    """Return a bounded snapshot and safe-to-display evidence for one local turn.

    Inputs are untrusted service snapshots. In particular, a remembered display
    identity is *not* a current identity until a fresh, consented face cue for
    exactly one matching front-camera track agrees with it.
    """
    now = time.time() if now is None else now
    snapshot = perception if isinstance(perception, dict) else {}
    fusion = snapshot.get("fusion") if isinstance(snapshot.get("fusion"), dict) else None
    fusion_items = [v for v in (fusion.get("items", []) if fusion else [])
                    if isinstance(v, dict)]

    def matching(kind, source, subject=None):
        # Affect belongs to a track, whereas presence/objects/posture are
        # source-level observations. A departed track must not veto the
        # current track, and another camera cannot supply its evidence.
        return [v for v in fusion_items if v.get("kind") == kind
                and v.get("source") == source and v.get("subject") == subject
                and v.get("session_id") is None]

    def allowed(kind, source, subject=None):
        if fusion is None:
            return True
        values = matching(kind, source, subject)
        return bool(fusion.get("ready") is not False and values
                    and all(v.get("state") == "present" and _fresh(v, now) for v in values))
    people = [p for p in (snapshot.get("people") or [])[:8] if _fresh(p, now)]
    people = [p for p in people if allowed("presence", p.get("source"))]
    front = [p for p in people if p.get("source") == "camera.front"]
    session_label = (identity.get("label") if isinstance(identity, dict)
                     and identity.get("status") == "known"
                     and identity.get("current_verified") is True else None)
    associated = front[0] if len(front) == 1 else None
    linked = associated.get("identity") if associated else None
    linked = linked if isinstance(linked, dict) else {}
    identity_ok = bool(request.get("input_type") != "voice" and session_label and linked.get("label") == session_label
                       and linked.get("consent") == "enrolled_local" and _fresh(linked, now)
                       and linked.get("source") == associated.get("source")
                       and linked.get("associated_track_id") == associated.get("track_id"))
    safe_people = [dict(_select(p, "track_id", "source", "confidence", "observed_at",
                                "expires_at", "provenance"),
                        identity=_select(p["identity"], "label", "consent", "confidence", "provenance",
                                         "source", "associated_track_id", "observed_at", "expires_at")
                        if identity_ok and p is associated else None) for p in people]
    objects = [_select(o, "label", "count", "confidence", "source", "observed_at",
                       "expires_at", "provenance") for o in (snapshot.get("objects") or [])[:20]
               if _fresh(o, now) and allowed("objects", o.get("source"))
               and (o.get("confidence") is None or o["confidence"] >= .4)]
    posture = snapshot.get("posture")
    posture = (_select(posture, "state", "reasons", "enabled", "calibrated", "confidence",
                       "source", "observed_at", "expires_at", "provenance")
               if _fresh(posture, now) and allowed("posture", posture.get("source")) else None)
    cues = [c for c in (snapshot.get("affect_cues") or [])[:2]
            if _fresh(c, now) and allowed("affect", c.get("source"), c.get("track_id"))
            and c.get("source") == "camera.front"
            and associated is not None and c.get("track_id") == associated.get("track_id")]
    cue = (_select(cues[0], "track_id", "source", "top_label", "confidence",
                   "observed_at", "expires_at", "provenance")
           if len(cues) == 1 and len(front) == 1 else None)
    messages = request.get("messages") if isinstance(request, dict) else None
    messages = messages if isinstance(messages, list) else []
    active_tasks = [
        {"state": t["state"], "summary": str(t.get("summary", ""))[:120]}
        for t in (tasks or [])[:8] if isinstance(t, dict)
        and t.get("state") in {"scheduled", "running"}
    ][:3]
    evidence = {
        "conversation": {"state": "available", "source": "conversation-session",
                         "turns": min(len(messages) // 2, 8), "observed_at": now},
        "identity": {"state": "verified" if identity_ok else "unconfirmed",
                     "source": "face-identity" if identity_ok else None},
        "people": _evidence(associated or (people[0] if people else None)),
        "objects": _evidence(objects[0] if objects else None),
        "posture": _evidence(posture),
        "affect": _evidence(cue, "available" if cue and cue.get("confidence", 0) >= .75 else "uncertain")
                  if cue else {"state": "missing"},
        "tasks": {"state": "available" if tasks is not None else "unavailable",
                  "source": "task-orchestrator" if tasks is not None else None,
                  "count": len(active_tasks), "observed_at": now if tasks is not None else None},
    }
    if fusion is not None:
        adopted = {"people": associated or (people[0] if people else None),
                   "objects": objects[0] if objects else None, "posture": posture, "affect": cue}
        for key, kind in {"people": "presence", "objects": "objects", "posture": "posture", "affect": "affect"}.items():
            item = adopted[key]
            if item is not None:
                matches = matching(kind, item.get("source"),
                                   item.get("track_id") if kind == "affect" else None)
            elif kind == "affect":
                # No unique front track means no association authority. Do
                # not advertise an old/other person's cue as used this turn.
                matches = (matching(kind, associated.get("source"), associated.get("track_id"))
                           if associated is not None else [])
            else:
                matches = [v for v in fusion_items if v.get("kind") == kind
                           and v.get("subject") is None and v.get("session_id") is None]
            if item is None:
                chosen = max(matches, key=lambda v: v.get("arrived_at", 0)) if matches else None
                state = chosen.get("state", "unknown") if chosen else "unknown"
                if chosen is not None and chosen.get("expires_at", 0) <= now:
                    state = "stale"
                elif state == "present":
                    # A live fusion record is not proof that its observation
                    # passed this turn's freshness, quality and association
                    # filters. Diagnostics describe what the prompt used.
                    state = "unknown"
                if fusion.get("ready") is False:
                    state = "unavailable"
                evidence[key] = {"state": state,
                                 "source": chosen.get("source") if chosen else None}
                if chosen is not None and chosen.get("state") == "present":
                    evidence[key]["reason"] = "no_eligible_observation"
            evidence[key]["evidence_refs"] = [r for v in matches for r in v.get("evidence", [])][:8]
        evidence["fusion"] = {"state": "available" if fusion.get("ready") else "unavailable",
                              "source": "fusion-core", "version": fusion.get("version"),
                              "mode": fusion.get("mode")}
    device_state = [{"kind": v["kind"], "state": v["state"], "data": v["data"],
                     "observed_at": v.get("observed_at"), "expires_at": v.get("expires_at")}
                    for v in fusion_items if v.get("kind") in {"interaction", "output", "resources"}
                    and v.get("source") != "output-events" and v.get("expires_at", 0) > now][:3]
    return {
        "captured_at": now, "perception_revision": snapshot.get("revision"),
        "identity_verified": identity_ok, "identity_label": session_label if identity_ok else None,
        "perception": {"observations": [_select(o, "kind", "provider", "source", "observed_at",
                                                "expires_at", "confidence", "inference",
                                                "observed_state", "confirmed_count")
                                        for o in (snapshot.get("observations") or [])[:32]
                                        if isinstance(o, dict) and o.get("expires_at", 0) > now],
                       "people": safe_people, "identities": [], "objects": objects,
                       "posture": posture, "affect_cues": [cue] if cue else []},
        "tasks": active_tasks, "evidence": evidence,
        "fusion_version": fusion.get("version") if fusion else None,
        "device_state": device_state,
        "recent_semantic_changes": _recent_changes(fusion, now),
    }
