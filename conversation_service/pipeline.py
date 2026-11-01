"""Adapter-neutral, atomic conversation turn orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TurnResult:
    session_id: str
    turn_id: str
    answer: str
    revision: int
    backend: str


class TurnCancelled(RuntimeError):
    pass


class ConversationPipeline:
    """Connect an input adapter to any responder through shared session state.

    ``responder`` receives a plain dictionary containing the current input,
    prior messages, and structured working memory.  It may call a local model,
    Hermes, a deterministic skill, or a cloud fallback.  UI and audio are
    intentionally outside this transaction.
    """

    def __init__(self, sessions, responder: Callable[[dict], str | tuple[str, str]], *,
                 backend: str | None = None, max_answer_chars: int = 1200):
        if backend is not None and not backend:
            raise ValueError("invalid conversation pipeline configuration")
        if max_answer_chars <= 0:
            raise ValueError("invalid conversation pipeline configuration")
        self.sessions = sessions
        self.responder = responder
        self.backend = backend
        self.max_answer_chars = max_answer_chars

    def run(self, session_id: str, user_text: str, *, input_type="text",
            is_active: Callable[[], bool] | None = None) -> TurnResult:
        started = self.sessions.start_turn(session_id, user_text, input_type=input_type)
        active = started["active_turn"]
        turn_id = active["turn_id"]
        request = {
            "session_id": session_id,
            "turn_id": turn_id,
            "user_text": active["user_text"],
            "input_type": active["input_type"],
            "messages": started.get("messages", []),
            "memory": started.get("memory", {}),
        }
        try:
            response = self.responder(request)
            if isinstance(response, tuple) and len(response) == 2:
                answer, backend = response
            else:
                answer, backend = response, self.backend
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError("responder returned an empty answer")
            if not isinstance(backend, str) or not backend.strip():
                raise RuntimeError("responder did not identify its backend")
            if is_active is not None and not is_active():
                self.sessions.cancel_turn(session_id, turn_id, reason="interrupted")
                raise TurnCancelled("interaction turn was interrupted")
            answer = answer.strip()
            if len(answer) > self.max_answer_chars:
                raise RuntimeError("responder answer exceeds conversation limit")
            completed = self.sessions.complete_turn(
                session_id, turn_id, answer, backend=backend,
            )
            return TurnResult(session_id, turn_id, answer,
                              completed["revision"], backend)
        except Exception as exc:
            if isinstance(exc, TurnCancelled):
                raise
            try:
                self.sessions.fail_turn(session_id, turn_id, reason=type(exc).__name__)
            except Exception:
                pass
            raise
