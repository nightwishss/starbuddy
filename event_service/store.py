"""Bounded in-memory semantic event ring; raw media never belongs here."""

from __future__ import annotations

import json
import re
import secrets
import threading
import time
from collections import deque
from copy import deepcopy


TOPIC = re.compile(r"^[a-z][a-z0-9]*(?:\.[a-z0-9_-]+){1,7}$")
SOURCE = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
PRIVACY = {"public", "local", "local-private", "sensitive"}
MAX_PAGE_BYTES = 512 * 1024  # Leave envelope headroom below the client's 1 MiB cap.


class EventError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class EventStore:
    def __init__(self, *, capacity: int = 512, clock=time.time):
        if not 16 <= capacity <= 4096:
            raise ValueError("capacity out of range")
        self.capacity = capacity
        self.clock = clock
        # Revision alone cannot distinguish a restarted bus that has already
        # advanced past a consumer's old cursor. The epoch is additive metadata;
        # existing numeric cursor clients retain their original API.
        self.epoch = secrets.token_hex(16)
        self.revision = 0
        self.events = deque(maxlen=capacity)
        self.event_sizes = deque(maxlen=capacity)
        self.condition = threading.Condition()

    @staticmethod
    def _topics(value) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, list) or len(value) > 32:
            raise EventError("invalid_topics", "topics must be a list")
        topics = tuple(value)
        if any(not isinstance(x, str) or not TOPIC.fullmatch(x) for x in topics):
            raise EventError("invalid_topics", "invalid topic")
        return topics

    def publish(self, *, topic: str, source: str, payload: dict,
                privacy: str = "local", trace_id: str | None = None) -> dict:
        if not isinstance(topic, str) or not TOPIC.fullmatch(topic):
            raise EventError("invalid_topic", "invalid topic")
        if not isinstance(source, str) or not SOURCE.fullmatch(source):
            raise EventError("invalid_source", "invalid source")
        if privacy not in PRIVACY:
            raise EventError("invalid_privacy", "invalid privacy label")
        if not isinstance(payload, dict):
            raise EventError("invalid_payload", "payload must be an object")
        payload = deepcopy(payload)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 16 * 1024:
            raise EventError("payload_too_large", "event payload exceeds 16 KiB")
        if trace_id is not None and (not isinstance(trace_id, str) or not SOURCE.fullmatch(trace_id)):
            raise EventError("invalid_trace_id", "invalid trace id")
        with self.condition:
            self.revision += 1
            event = {
                "epoch": self.epoch, "revision": self.revision, "topic": topic, "source": source,
                "privacy": privacy, "payload": payload, "created_at": self.clock(),
            }
            if trace_id is not None:
                event["trace_id"] = trace_id
            self.events.append(event)
            self.event_sizes.append(len(json.dumps(event, ensure_ascii=False,
                                                   separators=(",", ":")).encode("utf-8")))
            self.condition.notify_all()
            return deepcopy(event)

    def read(self, *, after_revision: int = 0, topics=None, limit: int = 64) -> dict:
        selected = self._topics(topics)
        if not isinstance(after_revision, int) or after_revision < 0:
            raise EventError("invalid_revision", "invalid revision")
        if not isinstance(limit, int) or not 1 <= limit <= 256:
            raise EventError("invalid_limit", "invalid limit")
        with self.condition:
            values, page_bytes, has_more = [], 0, False
            for event, size in zip(self.events, self.event_sizes):
                if event["revision"] <= after_revision or (selected and event["topic"] not in selected):
                    continue
                if len(values) >= limit or page_bytes + size + 1 > MAX_PAGE_BYTES:
                    has_more = True
                    break
                values.append(deepcopy(event))
                page_bytes += size + 1
            oldest = self.events[0]["revision"] if self.events else self.revision + 1
            return {"epoch": self.epoch, "revision": self.revision, "oldest_revision": oldest,
                    "reset": after_revision > self.revision,
                    "gap": bool(after_revision and after_revision < oldest - 1),
                    "events": values, "has_more": has_more,
                    "next_revision": values[-1]["revision"] if has_more else self.revision}

    def wait(self, *, after_revision: int, topics=None, timeout: float = 20.0,
             limit: int = 64) -> dict:
        if not isinstance(timeout, (int, float)) or not 0 <= timeout <= 30:
            raise EventError("invalid_timeout", "timeout out of range")
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                value = self.read(after_revision=after_revision, topics=topics, limit=limit)
                if value["events"] or value["gap"] or value["reset"]:
                    return value
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return value
                self.condition.wait(remaining)

    def stats(self) -> dict:
        with self.condition:
            return {"epoch": self.epoch, "revision": self.revision, "retained": len(self.events),
                    "capacity": self.capacity, "persistence": "none",
                    "raw_media_allowed": False}
