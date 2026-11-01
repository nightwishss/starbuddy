"""Finite, offline programmes above Output Arbiter; no model or audio bypass.

One in-memory player, at most one worker while playing/draining. Pause resumes
from the beginning of the current paragraph, not from a sample in the waveform.
"""
from __future__ import annotations
from gateway.speech.tempo import validate_speed

import hashlib
import re
import threading
import time
import uuid

PROGRAMS = {
    "moon-pocket": {
        "title": "月亮的小口袋", "mode": "bedtime", "symbol": "moon",
        "description": "三小段原创晚安故事，不着急睡着。",
        "segments": (
            "小怪兽有一只软软的口袋。今晚，它想把白天喜欢的小事，轻轻放进去。第一件，是小蜗牛送来的一片圆叶子。",
            "第二件，是它自己选的云朵帽。帽子安静地靠在枕头旁。第三件，还没有决定。小怪兽想，留一个空位也很好。",
            "窗外的月亮慢慢升高。小怪兽把口袋放好，轻轻说，今天就到这里。你也可以休息一会儿。不想马上睡着，也没关系。晚安。",
        ),
    },
    "seed": {
        "title": "小种子的旅行", "mode": "radio", "symbol": "leaf",
        "description": "听两段小知识，认识种子的旅行。",
        "segments": (
            "有些种子会旅行。蒲公英的种子带着轻轻的绒毛，风一吹，就可能飘到新的地方。它没有翅膀，却能搭风的便车。",
            "还有些植物的种子，会附着在动物的毛上，跟着走一段路。种子落到合适的地方，有了水和适宜的温度，就有机会发芽。",
        ),
    },
    "small-light": {
        "title": "一盏小小的灯", "mode": "radio", "symbol": "star",
        "description": "一首原创小诗，听完就停。",
        "segments": (
            "小小的灯，亮在窗边。不催花开，不催明天。小小的船，停在岸边。风也歇一歇，梦再慢慢编。",
        ),
    },
    "riddle": {
        "title": "猜一位慢朋友", "mode": "radio", "symbol": "shell",
        "description": "谜面后会暂停，想听答案再继续。",
        "segments": (
            "有位小朋友，背着小房子。走路慢慢的，下雨也不急。它可能是谁呢？不用着急回答，想听答案时，点继续。",
            "我的答案是蜗牛。它的壳像一座小房子。你想到别的答案也可以，我们可以一起说说为什么。",
        ),
        "pause_after": (0,),
    },
}
TERMINAL = {"completed", "cancelled", "interrupted", "failed", "expired"}
PAGE_LEASE_SECONDS = 20
PARAGRAPH_TIMEOUT_SECONDS = 180
SESSION_SECONDS = 30 * 60


class QuietError(RuntimeError):
    pass


def _owner(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("invalid owner")
    return hashlib.sha256(value.encode()).hexdigest()


def catalogue():
    return [{"id": key, **{k: item[k] for k in ("title", "mode", "symbol", "description")},
             "paragraphs": len(item["segments"]), "source": "本项目原创文本；无外部音频资产"}
            for key, item in PROGRAMS.items()]


class QuietPlayer:
    def __init__(self, output, *, busy=lambda: False, clock=time.monotonic, threaded=True):
        self.output, self.busy, self.clock, self.threaded = output, busy, clock, threaded
        self.lock = threading.RLock()
        self.io_lock = threading.Lock()
        self.wake = threading.Event()
        self.worker = None
        self.closed = False
        self.owner = self.session = self.program = None
        self.state, self.reason = "idle", None
        self.index = self.shown_index = self.revision = self.generation = 0
        self.job = None
        self.inflight = self.quarantined = self.cancel_sent = False
        self.errors = 0
        self.started = self.seen = self.job_started = 0.0
        self.cancel_deadline = None
        self.next_at = 0.0

    def _snapshot(self, owner):
        owned = self.owner == owner
        shown = self.shown_index
        return {"ok": True, "available": self.output is not None and not self.closed,
                "occupied": bool(self.quarantined or self.inflight or self.job or self.state in {"playing", "paused"}),
                "player": ({"id": self.session, "program_id": self.program,
                    "state": self.state, "reason": self.reason, "revision": self.revision,
                    "paragraph": min(shown + 1, len(PROGRAMS[self.program]["segments"])), "paragraphs": len(PROGRAMS[self.program]["segments"]),
                    "text": PROGRAMS[self.program]["segments"][min(shown, len(PROGRAMS[self.program]["segments"]) - 1)],
                    "pending_output": bool(self.job or self.inflight), "quarantined": self.quarantined,
                    "volume_percent": self.volume, "speed": self.speed,
                    "scheduled": self.scheduled} if owned else None)}

    def status(self, owner):
        key = _owner(owner)
        with self.lock:
            if key == self.owner:
                self.seen = self.clock()
            return self._snapshot(key)

    def start(self, owner, program_id, volume_percent, speed=1.0):
        return self._start(_owner(owner), program_id, volume_percent, speed)

    def _start(self, key, program_id, volume_percent, speed=1.0, *, scheduled=False):
        speed = validate_speed(speed)
        if (not isinstance(program_id, str) or program_id not in PROGRAMS
                or type(volume_percent) is not int or not 0 <= volume_percent <= 60):
            raise ValueError("invalid programme")
        with self.lock:
            if self.closed or self.output is None:
                raise QuietError("播放服务未就绪；不会绕过 Output 直接出声")
            if self.quarantined or self.job or self.inflight or self.state in {"playing", "paused"}:
                raise QuietError("请先停止当前节目，等待播放释放后再开始")
            self.owner, self.program, self.volume = key, program_id, volume_percent
            self.speed, self.scheduled = speed, scheduled
            self.session = "quiet_" + uuid.uuid4().hex
            self.index, self.shown_index, self.revision = 0, 0, 1
            self.generation += 1
            self.state, self.reason, self.errors = "playing", None, 0
            self.started = self.seen = self.next_at = self.clock()
            self._wake_locked()
            return self._snapshot(key)

    def control(self, owner, session_id, revision, action):
        key = _owner(owner)
        with self.lock:
            if key != self.owner or session_id != self.session:
                raise QuietError("这个节目不属于当前页面")
            if action not in {"pause", "resume", "next", "stop"} or type(revision) is not int:
                raise ValueError("invalid control")
            # Stop is idempotent and remains usable if a completion raced the UI.
            if action != "stop" and revision != self.revision:
                raise QuietError("节目进度已变化，请刷新后再试")
            if action != "stop" and (self.quarantined or self.state not in {"playing", "paused"}):
                raise QuietError("当前节目不能继续，请检查播放状态")
            if action == "resume" and (self.job or self.inflight):
                raise QuietError("上一段正在停止，请稍等")
            waiting_for_answer = self.state == "paused" and self.reason == "your_turn"
            self.seen = self.clock()
            self.generation += 1
            self.revision += 1
            self.reason = None
            if action == "stop":
                self.state, self.reason = "stopped", "user_stop"
                # Retry cleanup of a known job only; unknown submissions stay quarantined.
                if self.job:
                    self.quarantined, self.errors, self.cancel_sent = False, 0, False
                    self.cancel_deadline = None
            elif action == "pause":
                self.state, self.reason = "paused", "user_pause"
            elif action == "next":
                if not waiting_for_answer:
                    self.index += 1
                self.shown_index = self.index
                self.state = "playing" if self.index < len(PROGRAMS[self.program]["segments"]) else "completed"
            else:
                self.shown_index = self.index
                self.state = "playing"
            self.next_at = self.clock() + 1
            self._wake_locked()
            return self._snapshot(key)

    def interrupt(self):
        """Yield to foreground interaction; never automatically resume afterwards."""
        with self.lock:
            if self.state == "playing":
                self._pause_locked("conversation")
                self._wake_locked()

    def _pause_locked(self, reason):
        self.state, self.reason = "paused", reason
        self.generation += 1
        self.revision += 1

    def _wake_locked(self):
        self.wake.set()
        if self.threaded and (self.worker is None or not self.worker.is_alive()):
            self.worker = threading.Thread(target=self._run, name="vq-quiet-programme", daemon=True)
            self.worker.start()

    def _run(self):
        while True:
            self.wake.clear()
            self.tick()
            with self.lock:
                if self.quarantined or (not self.job and not self.inflight and self.state != "playing"):
                    self.worker = None
                    return
            self.wake.wait(1.0)

    def tick(self):
        """One bounded iteration, injectable clock/output for deterministic tests."""
        with self.io_lock:
            with self.lock:
                if self.quarantined:
                    return
                now = self.clock()
                if self.state == "playing":
                    if now - self.started >= SESSION_SECONDS:
                        self.state, self.reason = "stopped", "session_limit"
                        self.generation += 1
                    elif not self.scheduled and now - self.seen > PAGE_LEASE_SECONDS:
                        self._pause_locked("page_missing")
                    elif self.busy():
                        self._pause_locked("conversation")
                job = self.job
                if job:
                    cancel = self.generation != self.job_generation or now - self.job_started >= PARAGRAPH_TIMEOUT_SECONDS
                    if cancel and self.cancel_deadline is None:
                        self.cancel_deadline = now + 10
                    if self.cancel_deadline is not None and now >= self.cancel_deadline:
                        self.state, self.reason, self.quarantined = "error", "cleanup_unconfirmed", True
                        return
                    send_cancel = cancel and not self.cancel_sent
                elif self.closed or self.state != "playing" or now < self.next_at:
                    return
                else:
                    send_cancel = False
            try:
                if job:
                    if send_cancel:
                        self.output.cancel(job)
                        with self.lock:
                            self.cancel_sent = True
                    result = self.output.status(job)
                    with self.lock:
                        self.errors = 0
                        if (result.get("state") == "queued" and self.state == "playing"
                                and self.clock() - self.job_started > 5):
                            self._pause_locked("output_busy")
                        if result.get("state") in TERMINAL and result.get("ended_at") is not None:
                            same = self.generation == self.job_generation
                            self.job, self.cancel_sent, self.cancel_deadline = None, False, None
                            if same and self.state == "playing":
                                if result["state"] == "completed":
                                    old_index = self.index
                                    self.index += 1
                                    self.revision += 1
                                    if self.index >= len(PROGRAMS[self.program]["segments"]):
                                        self.state = "completed"
                                    elif old_index in PROGRAMS[self.program].get("pause_after", ()):
                                        self._pause_locked("your_turn")
                                    else:
                                        self.shown_index = self.index
                                    self.next_at = self.clock() + 2
                                else:
                                    self._pause_locked("output_" + result["state"])
                    return
                # Low priority is enforced by the existing arbiter as well as
                # this preflight. Do not queue programmes behind other output.
                system = self.output.system_status()
                with self.lock:
                    if self.state != "playing" or self.closed:
                        return
                    if not system.get("ready") or system.get("current") or system.get("queued"):
                        self._pause_locked("output_busy")
                        return
                    self.inflight = True
                    generation, session = self.generation, self.session
                    text = PROGRAMS[self.program]["segments"][self.index]
                    volume = self.volume
                try:
                    result = self.output.say(text, owner="session:" + session, category="notification",
                                             volume_percent=volume, expires_in_seconds=PARAGRAPH_TIMEOUT_SECONDS,
                                             trace_id=session, **({"speed": self.speed} if self.speed != 1 else {}))
                    job_id = result.get("job_id")
                    if not isinstance(job_id, str) or not job_id:
                        raise ValueError("missing job id")
                except Exception:
                    # A transport timeout may have accepted the submission.
                    # Never retry it and accidentally speak the same text twice.
                    with self.lock:
                        self.state, self.reason, self.quarantined = "error", "submission_unknown", True
                    return
                finally:
                    with self.lock:
                        self.inflight = False
                with self.lock:
                    self.job, self.job_generation = job_id, generation
                    self.job_started = self.clock()
                    self.cancel_sent, self.cancel_deadline, self.errors = False, None, 0
                    if generation != self.generation:
                        self.wake.set()
            except Exception:
                with self.lock:
                    self.errors += 1
                    if self.errors >= 3:
                        self.state, self.reason = "error", "output_unavailable"
                        self.quarantined = bool(self.job)

    def close(self):
        with self.lock:
            self.closed = True
            self.state, self.reason = "stopped", "service_exit"
            self.generation += 1
            if self.job or self.inflight:
                self._wake_locked()
