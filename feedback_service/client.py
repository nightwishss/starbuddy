"""Dependency-free client for the local Feedback Collector."""

from __future__ import annotations

import json
import os
import socket


DEFAULT_SOCKET = os.environ.get("VQ_FEEDBACK_SOCKET", f"/run/user/{os.getuid()}/vq-feedback.sock")


class FeedbackClientError(RuntimeError):
    def __init__(self, code, message): super().__init__(message); self.code = code


class FeedbackClient:
    def __init__(self, socket_path=DEFAULT_SOCKET, *, timeout=5):
        self.socket_path, self.timeout = socket_path, timeout
    def call(self, operation, **values):
        body=json.dumps({"operation":operation,**values},ensure_ascii=False,separators=(",", ":"))+"\n"
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout); client.connect(self.socket_path); client.sendall(body.encode())
            raw=client.makefile("rb").readline(1024*1024)
        try: result=json.loads(raw)
        except (json.JSONDecodeError,UnicodeDecodeError) as exc:
            raise FeedbackClientError("invalid_response","invalid Feedback response") from exc
        if not result.get("ok"):
            raise FeedbackClientError(result.get("code","service_error"),result.get("error","Feedback error"))
        return result["result"]
    def label_current_posture(self,owner,label,**options):
        return self.call("posture.label_current",owner=owner,label=label,**options)
    def label_vision_summary(self,owner,counts,human_present,confidence,verdict,**options):
        return self.call("vision.label_summary",owner=owner,counts=counts,human_present=human_present,
                         confidence=confidence,verdict=verdict,**options)
    def rate_conversation(self,model_id,rating,**options):
        return self.call("conversation.rate",model_id=model_id,rating=rating,**options)
    def stats(self): return self.call("stats")
