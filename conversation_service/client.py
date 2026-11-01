"""Small dependency-free client for the local Conversation Session service."""

from __future__ import annotations

import json
import os
import socket


DEFAULT_SOCKET = os.environ.get("VQ_CONVERSATION_SOCKET", "/run/user/1000/vq-conversation.sock")


class ConversationClientError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class ConversationClient:
    def __init__(self, socket_path: str = DEFAULT_SOCKET, *, timeout: float = 35.0):
        self.socket_path = socket_path
        self.timeout = timeout

    def call(self, operation: str, **arguments) -> dict:
        request = json.dumps({"operation": operation, **arguments}, ensure_ascii=False) + "\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout)
            client.connect(self.socket_path)
            client.sendall(request.encode("utf-8"))
            response = bytearray()
            while not response.endswith(b"\n"):
                chunk = client.recv(65536)
                if not chunk:
                    break
                response.extend(chunk)
                if len(response) > 1024 * 1024:
                    raise ConversationClientError("response_too_large", "conversation response too large")
        try:
            payload = json.loads(response)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ConversationClientError("invalid_response", "invalid conversation response") from exc
        if not payload.get("ok"):
            raise ConversationClientError(payload.get("code", "service_error"),
                                          payload.get("error", "conversation service error"))
        return payload["result"]

    def ensure(self, session_id: str, *, privacy="local-private") -> dict:
        return self.call("ensure", session_id=session_id, privacy=privacy)

    def context(self, session_id: str) -> dict:
        return self.call("context", session_id=session_id)

    def start_turn(self, session_id: str, user_text: str, *, input_type="text") -> dict:
        return self.call("turn.start", session_id=session_id, user_text=user_text,
                         input_type=input_type)

    def complete_turn(self, session_id: str, turn_id: str, assistant_text: str,
                      *, backend="unknown") -> dict:
        return self.call("turn.complete", session_id=session_id, turn_id=turn_id,
                         assistant_text=assistant_text, backend=backend)

    def fail_turn(self, session_id: str, turn_id: str, *, reason=None) -> dict:
        return self.call("turn.fail", session_id=session_id, turn_id=turn_id, reason=reason)

    def cancel_turn(self, session_id: str, turn_id: str, *, reason=None) -> dict:
        return self.call("turn.cancel", session_id=session_id, turn_id=turn_id, reason=reason)

    def remember(self, session_id: str, key: str, value, *, source: str,
                 confidence=1.0, ttl_seconds=None) -> dict:
        values = {"session_id": session_id, "key": key, "value": value,
                  "source": source, "confidence": confidence}
        if ttl_seconds is not None:
            values["ttl_seconds"] = ttl_seconds
        return self.call("memory.set", **values)

    def forget(self, session_id: str, key: str) -> dict:
        return self.call("memory.forget", session_id=session_id, key=key)

    def reset(self, session_id: str) -> dict:
        return self.call("reset", session_id=session_id)

    def wait(self, session_id: str, after_revision: int, *, timeout=20.0) -> dict:
        return self.call("wait", session_id=session_id, after_revision=after_revision,
                         timeout=timeout)

    def stats(self) -> dict:
        return self.call("stats")
