"""Dedicated owner for posture baseline, temporal policy, events and reminders."""

from __future__ import annotations

import argparse
import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from camera_stream_service import CameraStreamClient
from event_service import EventClient
from gateway.display.pose_client import PoseClient
from gateway.display.server import POSTURE_OWNER, PostureController
from output_service import OutputClient
from runtime.client import RuntimeClient
from sdk.audio import AudioClient
from sdk.speech import SpeechClient
from perception_context_service.client import PerceptionContextClient
from .lease import PostureLeaseError, PostureMonitorLease


class FusionReminderCallbacks:
    """Carry one admission ticket to its submit on the same reminder worker.

    A proposal reserves the shared ambient slot; asking for another one at
    submit time would reject our own reservation. The ticket is consumed
    before Output is called, including when the submission reply is lost.
    Output still revalidates the original ticket before any audible effect.
    """

    def __init__(self, context, output, mode):
        self.context, self.output, self.mode = context, output, mode
        self.pending = threading.local()

    def admit(self):
        # Never let a failed/new admission reuse an earlier worker ticket.
        self.pending.guard = None
        self.pending.decision = {"allowed": False, "reason": "admission_unknown"}
        if self.mode != "enabled":
            self.pending.decision = {"allowed": True, "reason": "posture_reminder_ready"}
            return True
        try:
            proposal = self.context.posture_proposal()
            guard = proposal.get("guard")
            decision = proposal.get("decision")
            decision = decision if isinstance(decision, dict) else {}
            reason = decision.get("reason")
            if isinstance(reason, str) and reason:
                self.pending.decision["reason"] = reason[:96]
            if (proposal.get("mode") != "enabled"
                    or decision.get("action") != "propose"
                    or not isinstance(guard, dict) or set(guard) != {"kind", "token"}
                    or guard.get("kind") != "posture"
                    or not isinstance(guard.get("token"), str) or not guard["token"]):
                return False
        except Exception:
            self.pending.decision = {"allowed": False, "reason": "fusion_unavailable"}
            return False
        self.pending.guard = dict(guard)
        self.pending.decision = {"allowed": True, "reason": "posture_reminder_ready"}
        return True

    def diagnostic(self):
        """Report this worker's last admission, never reserve a second ticket."""
        return dict(getattr(self.pending, "decision", {
            "allowed": False, "reason": "admission_unknown",
        }))

    def remind(self, text, owner, volume):
        guard = None
        if self.mode == "enabled":
            guard = getattr(self.pending, "guard", None)
            self.pending.guard = None
            if guard is None:
                return None
        elif self.mode == "shadow":
            # Preserve shadow diagnostics without taking an effective ticket.
            try:
                self.context.posture_proposal()
            except Exception:
                pass
        return self.output.say(text, owner, category="notification",
                               volume_percent=volume, expires_in_seconds=15,
                               guard=guard)["job_id"]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args): pass
    @property
    def posture(self): return self.server.posture
    @property
    def monitor(self): return self.server.monitor
    def do_GET(self):
        if urlparse(self.path).path == "/api/posture/status": return self._json(200,self.monitor.status())
        if urlparse(self.path).path == "/healthz":
            value=self.posture.status(); return self._json(200 if value.get("pose") else 503,{"ready":bool(value.get("pose"))})
        self._json(404,{"error":"not_found"})
    def do_POST(self):
        path=urlparse(self.path).path
        operations={"/api/posture/start":"start","/api/posture/pause":"pause",
                    "/api/posture/calibrate":"calibrate","/api/posture/reset":"reset",
                    "/api/posture/simulate":"simulate","/api/posture/monitor":"monitor",
                    "/api/posture/stop":"stop"}
        if path not in operations: return self._json(404,{"error":"not_found"})
        if self.headers.get_content_type() != "application/json" or int(self.headers.get("Content-Length","0"))>1024:
            return self._json(400,{"error":"invalid_request"})
        raw=self.rfile.read(int(self.headers.get("Content-Length","0")))
        try: body=json.loads(raw or b"{}")
        except (ValueError, json.JSONDecodeError): return self._json(400,{"error":"invalid_request"})
        if not isinstance(body,dict): return self._json(400,{"error":"invalid_request"})
        try:
            operation=operations[path]
            if operation=="monitor": value=self.monitor.monitor(**body)
            elif operation=="stop": value=self.monitor.stop(**body)
            elif body: return self._json(400,{"error":"invalid_request"})
            elif operation=="start": value=self.monitor.manual_start()
            elif operation=="pause": value=self.monitor.manual_pause()
            else: value=getattr(self.posture,operation)()
        except (PostureLeaseError,TypeError) as exc: return self._json(409,{"error":str(exc)[:120]})
        except Exception as exc: return self._json(503,{"error":type(exc).__name__})
        self._json(200,value)
    def _json(self,status,value):
        body=json.dumps(value,ensure_ascii=False,separators=(",", ":")).encode()
        self.send_response(status); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(body))); self.send_header("Cache-Control","no-store")
        self.end_headers(); self.wfile.write(body)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--host",default="127.0.0.1"); parser.add_argument("--port",type=int,default=8792)
    args=parser.parse_args()
    if args.host not in {"127.0.0.1","::1"}: parser.error("posture service is loopback only")
    runtime=RuntimeClient(timeout=35); speech=SpeechClient(runtime); audio=AudioClient(runtime)
    output=OutputClient(timeout=35); events=EventClient(timeout=1)
    output_status=OutputClient(timeout=.15)
    context = PerceptionContextClient(timeout=.8)
    fusion_mode = os.environ.get("VQ_CONTEXT_POSTURE_MODE", "shadow")
    if fusion_mode not in {"off", "shadow", "enabled"}:
        parser.error("invalid VQ_CONTEXT_POSTURE_MODE")
    reminders = FusionReminderCallbacks(context, output, fusion_mode)
    posture=PostureController(
        speech,audio,CameraStreamClient(consumer="posture",rotation=180),PoseClient(),
        reminder_player=reminders.remind,reminder_stopper=lambda job_id: output.cancel(job_id),
        reminder_status=lambda job_id: output_status.status(job_id),
        reminder_admission=reminders.admit,
        baseline_path=os.environ.get("VQ_POSTURE_BASELINE_PATH",str(
            Path.home()/".local/state/vq-platform/posture-baseline.json")),
        events=events,
        # A reminder already has a confirmed posture-policy decision. Use its
        # short gentle fallback without reserving the language model, in every
        # fusion mode; background phrasing must never delay a real dialogue or
        # make the fresh pose expire while waiting for Qwen.
        reminder_generator=None)
    monitor=PostureMonitorLease(
        posture, Path(os.environ.get("VQ_POSTURE_LEASE_PATH",str(
            Path.home()/".local/state/vq-platform/posture-monitor.json"))),
        events=events, autostart=os.environ.get("VQ_POSTURE_AUTOSTART","1") == "1",
    )
    server=ThreadingHTTPServer((args.host,args.port),Handler); server.daemon_threads=True
    server.posture=posture; server.monitor=monitor
    try: server.serve_forever()
    finally: monitor.close(); posture.close()

if __name__ == "__main__": main()
