"""S5: stateless private listening and an explicitly authored, locked local diary.

No Conversation, model, Event Bus, Output or personal-memory dependencies.
This is a local appliance lock, not encrypted storage or a multi-user account system.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import stat
import threading
import time
from contextlib import closing
from pathlib import Path

from gateway.display.story import PRIVATE
from governance.privacy import DataLabel
from governance.safety import ChildSafetyPolicy

SESSION_TTL = 300
RETENTION = 30 * 86400
MAX_ENTRIES = 100
MOODS = {"unknown": "没有选择", "happy": "开心", "calm": "平静", "sad": "难过",
         "worried": "担心", "angry": "生气", "mixed": "说不清 / 混合"}
FIELDS = {
    "listen": {"text", "mode"}, "status": set(),
    "create": {"passphrase"}, "unlock": {"passphrase"}, "lock": {"token"},
    "list": {"token"}, "preview": {"token", "text", "mood"},
    "discard": {"token"}, "save": {"token", "draft_id", "consent"},
    "delete": {"token", "entry_id", "consent"},
}


class JournalError(RuntimeError):
    """Only authored, content-free messages may cross the HTTP boundary."""


def _text(value, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "\x00" in value:
        raise ValueError("invalid text")
    return value.strip()


def listen(text, mode):
    """The input lives only in this call; never echoed, stored, or sent to a model."""
    text = _text(text, 1000)
    if mode not in {"listen", "comfort"}:
        raise ValueError("invalid mode")
    safety = ChildSafetyPolicy().evaluate(text)
    if not safety.allow_model:
        answer = safety.response
    elif mode == "comfort":
        answer = "谢谢你愿意说出来。你可以慢慢来，不必马上解决。想休息一下，或者找信任的人陪一会儿，都可以。"
    else:
        answer = "我收到你写的话了。这里先陪你安静一会儿，你不需要解释更多。你可以继续写，也可以让这些字离开屏幕。"
    return {"answer": answer, "label": DataLabel.CHILD_PRIVATE.value, "stored": False,
            "response_kind": "authored_support", "model_used": False, "local_only": True}


class Journal:
    def __init__(self, root=None, *, clock=time.monotonic, wall=time.time):
        self.root = Path(root) if root is not None else Path.home() / ".local/state/vq-platform/diary"
        self.path = self.root / "library.sqlite3"
        self.clock, self.wall = clock, wall
        self.lock = threading.RLock()
        self.sessions = {}
        self.attempts = []

    @staticmethod
    def _safe(path, directory=False):
        info = path.lstat()
        if (stat.S_ISLNK(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077 or
                not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode))):
            raise JournalError("日记文件权限不安全，请由设备管理员检查")

    def _db(self):
        # No implicit creation from treehole or status requests.
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._safe(self.root, directory=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        self._safe(self.path)
        db = sqlite3.connect(self.path, timeout=2)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA secure_delete=ON")
        db.execute("CREATE TABLE IF NOT EXISTS notebook (id INTEGER PRIMARY KEY CHECK(id=1), salt BLOB, digest BLOB)")
        db.execute("CREATE TABLE IF NOT EXISTS entries (id TEXT PRIMARY KEY, text TEXT, mood TEXT, created REAL, expires REAL)")
        db.execute("DELETE FROM entries WHERE expires <= ?", (self.wall(),))
        db.commit()
        return db

    @staticmethod
    def _digest(passphrase, salt):
        if not isinstance(passphrase, str) or not 6 <= len(passphrase) <= 64:
            raise JournalError("请使用 6–64 个字符的口令，建议使用较长短语")
        return hashlib.scrypt(passphrase.encode(), salt=salt, n=16384, r=8, p=1,
                              maxmem=64 * 1024 * 1024, dklen=32)

    def _prune(self):
        now = self.clock()
        self.sessions = {k: v for k, v in self.sessions.items() if now < v["expires"]}
        self.attempts = [t for t in self.attempts if now - t < 60]

    def _session(self, token):
        if not isinstance(token, str) or token not in self.sessions:
            raise JournalError("日记已锁定，请重新解锁")
        return self.sessions[token]

    def call(self, action, data):
        if action not in FIELDS or not isinstance(data, dict) or set(data) != FIELDS[action]:
            raise ValueError("invalid journal request")
        if action == "listen":
            return listen(**data)
        with self.lock:
            self._prune()
            if action == "status":
                if not self.path.exists():
                    return {"configured": False, "retention_days": 30}
                with closing(self._db()) as db:
                    return {"configured": db.execute("SELECT 1 FROM notebook").fetchone() is not None,
                            "retention_days": 30}
            if action in {"create", "unlock"}:
                if len(self.attempts) >= 5:
                    raise JournalError("尝试较多，请稍等一分钟再解锁")
                self.attempts.append(self.clock())
                if len(self.sessions) >= 8:
                    raise JournalError("打开的日记过多，请稍后再试")
                if action == "unlock" and not self.path.exists():
                    raise JournalError("请先设置这台板子的日记口令")
                with closing(self._db()) as db:
                    row = db.execute("SELECT salt, digest FROM notebook WHERE id=1").fetchone()
                    if action == "create":
                        if row:
                            raise JournalError("日记已经设置口令，请解锁")
                        salt = secrets.token_bytes(16)
                        digest = self._digest(data["passphrase"], salt)
                        db.execute("INSERT INTO notebook VALUES (1, ?, ?)", (salt, digest))
                        db.commit()
                    elif not row or not hmac.compare_digest(self._digest(data["passphrase"], row["salt"]), row["digest"]):
                        raise JournalError("口令不正确")
                token = secrets.token_hex(32)
                self.sessions[token] = {"expires": self.clock() + SESSION_TTL, "draft": None, "receipt": None}
                return {"token": token, "expires_in": SESSION_TTL}
            if action == "lock":
                if not isinstance(data["token"], str):
                    raise ValueError("invalid token")
                self.sessions.pop(data["token"], None)
                return {"locked": True}
            session = self._session(data["token"])
            if action == "discard":
                session["draft"] = None
                return {"discarded": True}
            if action == "preview":
                text = _text(data["text"], 1000)
                if PRIVATE.search(text):
                    raise JournalError("这段话标记了秘密或不保存，请留在树洞，不写入日记")
                if not isinstance(data["mood"], str) or data["mood"] not in MOODS:
                    raise ValueError("invalid mood")
                draft = {"id": secrets.token_hex(16), "text": text, "mood": data["mood"],
                         "source": "manual_diary", "label": DataLabel.PERSONAL.value}
                session["draft"], session["receipt"] = draft, None
                return {"draft": dict(draft)}
            with closing(self._db()) as db:
                if action == "list":
                    entries = [dict(row, source="manual_diary", mood_source="self_report") for row in
                               db.execute("SELECT * FROM entries ORDER BY created DESC, id LIMIT ?", (MAX_ENTRIES,))]
                    return {"entries": entries, "retention_days": 30}
                if data["consent"] is not True:
                    raise JournalError("需要明确确认，才会保存或删除")
                if action == "delete":
                    entry_id = data["entry_id"]
                    if not isinstance(entry_id, str) or len(entry_id) != 32:
                        raise ValueError("invalid entry")
                    db.execute("DELETE FROM entries WHERE id=?", (entry_id,))
                    db.commit()
                    session["receipt"] = None
                    return {"deleted": True}
                draft = session["draft"]
                if session["receipt"] and data["draft_id"] == session["receipt"]:
                    return {"saved": True, "entry_id": session["receipt"], "duplicate": True}
                if not draft or data["draft_id"] != draft["id"]:
                    raise JournalError("草稿已失效，请重新预览")
                if db.execute("SELECT COUNT(*) FROM entries").fetchone()[0] >= MAX_ENTRIES:
                    raise JournalError("日记已满，请先删除不需要的记录")
                now = self.wall()
                db.execute("INSERT INTO entries VALUES (?, ?, ?, ?, ?)",
                           (draft["id"], draft["text"], draft["mood"], now, now + RETENTION))
                db.commit()
                session["receipt"], session["draft"] = draft["id"], None
                return {"saved": True, "entry_id": draft["id"], "duplicate": False}
