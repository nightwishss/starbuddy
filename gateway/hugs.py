"""Small shared inbox for scoped, local-network hugs. Only preset IDs, no media."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import stat
import time
from contextlib import contextmanager
from pathlib import Path

TOPIC = "interaction.hug.requested"
PRESETS = {
    "hug": "有人给你送来一个轻轻的拥抱。你可以慢慢来。",
    "cheer": "有人在为你加油。按自己的节奏，试一小步就好。",
    "goodnight": "有人送来晚安问候。愿你今晚安安稳稳。",
}
LABELS = {"hug": "轻轻抱抱", "cheer": "为你加油", "goodnight": "晚安问候"}
ID = re.compile(r"^[a-f0-9]{32}$")
DEMO_OWNER = hashlib.sha256(b"local-hug-demo").hexdigest()
TERMINAL = {"completed", "silent", "busy", "expired", "cancelled", "failed", "unknown"}


class HugError(RuntimeError):
    pass


def owner_key(token):
    return hashlib.sha256(token.encode()).hexdigest()


class HugInbox:
    """Owner-only SQLite mailbox reused by Display, Phone Gateway and event consumer.

    No background threads; all private content systems are deliberately absent.
    One active request, 20s cooldown, 30s delivery TTL, 256 receipts / 24h maximum.
    """
    def __init__(self, root=None, *, clock=time.time):
        self.root = Path(root) if root else Path.home() / ".local/state/vq-platform/hugs"
        self.path = self.root / "inbox.sqlite3"
        self.clock = clock

    @contextmanager
    def db(self):
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        for path in (self.root,):
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise HugError("拥抱存储权限不安全")
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        os.close(fd)
        info = self.path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise HugError("拥抱存储权限不安全")
        db = sqlite3.connect(self.path, timeout=1)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE IF NOT EXISTS settings (id INTEGER PRIMARY KEY, enabled INTEGER, muted INTEGER, generation INTEGER, code TEXT, code_until REAL)")
            db.execute("INSERT OR IGNORE INTO settings VALUES (1,0,1,0,'',0)")
            db.execute("CREATE TABLE IF NOT EXISTS grants (owner TEXT PRIMARY KEY, generation INTEGER, expires REAL)")
            db.execute("CREATE TABLE IF NOT EXISTS hugs (id TEXT PRIMARY KEY, owner TEXT, kind TEXT, state TEXT, created REAL, expires REAL, job TEXT, visual INTEGER DEFAULT 0)")
            now = self.clock()
            db.execute("DELETE FROM grants WHERE expires <= ?", (now,))
            db.execute("DELETE FROM hugs WHERE created < ?", (now - 86400,))
            db.execute("UPDATE hugs SET state='expired' WHERE state='queued' AND expires<=?", (now,))
            db.execute("UPDATE hugs SET state='unknown' WHERE state='claimed' AND expires<=?", (now,))
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def _settings(db):
        return db.execute("SELECT * FROM settings WHERE id=1").fetchone()

    def _authorized(self, db, owner):
        settings = self._settings(db)
        return db.execute("SELECT 1 FROM grants WHERE owner=? AND generation=? AND expires>?",
                          (owner, settings['generation'], self.clock())).fetchone() is not None

    def authorize(self, owner):
        with self.db() as db:
            return self._authorized(db, owner)

    def pair(self, code, owner):
        if not isinstance(code, str) or not re.fullmatch(r"[0-9]{6}", code):
            return False
        with self.db() as db:
            settings = self._settings(db)
            if (not settings['enabled'] or settings['code_until'] <= self.clock() or not settings['code']
                    or not hmac.compare_digest(settings['code'], owner_key(code))):
                return False
            if db.execute("SELECT COUNT(*) FROM grants").fetchone()[0] >= 4:
                raise HugError("已授权设备已满，请先在板端撤销授权")
            db.execute("INSERT INTO grants VALUES (?,?,?)", (owner, settings['generation'], self.clock()+3600))
            db.execute("UPDATE settings SET code='',code_until=0 WHERE id=1")
            return True

    def control(self, action, data):
        fields = {"status": set(), "invite": set(), "revoke": set(), "settings": {"enabled", "muted"}}
        if action not in fields or set(data) != fields[action]:
            raise ValueError("invalid hug control")
        with self.db() as db:
            if action == "settings":
                if any(type(data[k]) is not bool for k in data):
                    raise ValueError("invalid setting")
                db.execute("UPDATE settings SET enabled=?,muted=? WHERE id=1", (data['enabled'], data['muted']))
                if not data['enabled']:
                    db.execute("UPDATE hugs SET state='cancelled' WHERE state='queued'")
                    db.execute("UPDATE settings SET code='',code_until=0 WHERE id=1")
            elif action == "revoke":
                db.execute("UPDATE settings SET enabled=0,generation=generation+1,code='',code_until=0 WHERE id=1")
                db.execute("DELETE FROM grants")
                db.execute("UPDATE hugs SET state='cancelled' WHERE state='queued'")
            elif action == "invite":
                if not self._settings(db)['enabled']:
                    raise HugError("请先在板端开启接收")
                code = f"{secrets.randbelow(1000000):06d}"
                db.execute("UPDATE settings SET code=?,code_until=? WHERE id=1", (owner_key(code), self.clock()+120))
                return {"code": code, "expires_in": 120}
            settings = self._settings(db)
            latest = db.execute("SELECT * FROM hugs ORDER BY created DESC LIMIT 1").fetchone()
            return {"enabled": bool(settings['enabled']), "muted": bool(settings['muted']),
                    "paired_devices": db.execute("SELECT COUNT(*) FROM grants").fetchone()[0],
                    "latest": self._public(latest) if latest else None}

    @staticmethod
    def _public(row):
        return {k: row[k] for k in ('id', 'kind', 'state', 'created', 'expires', 'visual')}

    def submit(self, owner, request_id, kind, events, *, demo=False):
        if not isinstance(request_id, str) or not ID.fullmatch(request_id) or not isinstance(kind, str) or kind not in PRESETS:
            raise ValueError("invalid hug")
        with self.db() as db:
            if not (demo and owner == DEMO_OWNER) and not self._authorized(db, owner):
                raise HugError("授权已失效，请重新配对")
            previous = db.execute("SELECT * FROM hugs WHERE id=?", (request_id,)).fetchone()
            if previous:
                if previous['owner'] != owner or previous['kind'] != kind:
                    raise HugError("请求编号已使用")
                return self._public(previous)
            if not self._settings(db)['enabled']:
                raise HugError("板端正在勿扰，不接收新拥抱")
            now = self.clock()
            if db.execute("SELECT 1 FROM hugs WHERE created>? OR state IN ('queued','claimed')", (now-20,)).fetchone():
                raise HugError("请稍等，拥抱之间至少间隔 20 秒")
            if db.execute("SELECT COUNT(*) FROM hugs").fetchone()[0] >= 256:
                raise HugError("今日拥抱记录已满，请明天再试")
            db.execute("INSERT INTO hugs VALUES (?,?,?,?,?,?,?,?)", (request_id,owner,kind,'queued',now,now+30,None,0))
        try:
            events.publish(TOPIC, 'phone-hug' if not demo else 'display-hug', {'hug_id':request_id}, privacy='local', trace_id='hug:'+request_id)
        except Exception:
            self.finish(request_id, 'failed')
            raise HugError("事件通道暂时不可用，这次不会自动重播") from None
        return self.receipt(owner, request_id, demo=demo)

    def receipt(self, owner, request_id, *, demo=False):
        if not isinstance(request_id, str) or not ID.fullmatch(request_id):
            raise ValueError("invalid receipt")
        with self.db() as db:
            if not (demo and owner == DEMO_OWNER) and not self._authorized(db, owner):
                raise HugError("授权已失效，请重新配对")
            row = db.execute("SELECT * FROM hugs WHERE id=? AND owner=?", (request_id,owner)).fetchone()
            if not row: raise HugError("没有这次拥抱的回执")
            return self._public(row)

    def claim(self, request_id):
        with self.db() as db:
            row = db.execute("SELECT * FROM hugs WHERE id=?", (request_id,)).fetchone()
            if not row or row['state'] != 'queued': return None
            settings = self._settings(db)
            if not settings['enabled'] or (row['owner'] != DEMO_OWNER and not self._authorized(db,row['owner'])):
                db.execute("UPDATE hugs SET state='cancelled' WHERE id=?", (request_id,)); return None
            db.execute("UPDATE hugs SET state='claimed' WHERE id=?", (request_id,))
            return dict(row, muted=bool(settings['muted']))

    def valid(self, request_id, *, audible=False):
        with self.db() as db:
            row = db.execute("SELECT * FROM hugs WHERE id=?", (request_id,)).fetchone()
            settings = self._settings(db)
            return bool(row and row['state']=='claimed' and row['expires']>self.clock() and settings['enabled']
                        and (not audible or not settings['muted'])
                        and (row['owner']==DEMO_OWNER or self._authorized(db,row['owner'])))

    def mark(self, request_id, *, visual=False, job=None):
        with self.db() as db:
            db.execute("UPDATE hugs SET visual=MAX(visual,?),job=COALESCE(?,job) WHERE id=? AND state='claimed'",
                       (int(visual),job,request_id))

    def finish(self, request_id, state):
        if state not in TERMINAL: raise ValueError("invalid hug state")
        with self.db() as db:
            db.execute("UPDATE hugs SET state=? WHERE id=? AND state IN ('queued','claimed')", (state,request_id))


class HugDelivery:
    """Runs in the existing event consumer; no new daemon or idle polling."""
    def __init__(self, inbox, display, output, *, sleep=time.sleep, clock=time.monotonic):
        self.inbox,self.display,self.output,self.sleep,self.clock = inbox,display,output,sleep,clock
        from output_service.client import OutputClient
        if isinstance(output, OutputClient):
            self.output = OutputClient(output.socket_path, timeout=1)

    def deliver(self, request_id):
        row = self.inbox.claim(request_id)
        if row is None: return
        trace = 'hug:'+request_id
        job = None
        try:
            # Fail closed when output readiness is unknown, even for silent hugs.
            status = self.output.system_status()
            if not status.get('ready') or status.get('current') or status.get('queued'):
                self.inbox.finish(request_id,'busy'); return
            if not self.inbox.valid(request_id):
                self.inbox.finish(request_id,'cancelled'); return
            shown = self.display.update(producer='task',phase='attention',visual_style='gentle',
                                        primary_text=LABELS[row['kind']],secondary_text=PRESETS[row['kind']],trace_id=trace)
            if shown.get('trace_id') != trace:
                self.inbox.finish(request_id,'busy'); return
            self.inbox.mark(request_id,visual=True)
            if row['muted']:
                self.sleep(2)
                self.inbox.finish(request_id,'silent' if self.inbox.valid(request_id) else 'cancelled'); return
            if not self.inbox.valid(request_id,audible=True):
                self.inbox.finish(request_id,'cancelled'); return
            job = self.output.say(PRESETS[row['kind']],trace,category='notification',volume_percent=55,
                                  expires_in_seconds=max(1, min(20, row['expires']-self.inbox.clock())),trace_id=trace)['job_id']
            self.inbox.mark(request_id,job=job)
            deadline = self.clock()+25
            while self.clock()<deadline:
                if not self.inbox.valid(request_id,audible=True):
                    self.output.cancel(job); self.inbox.finish(request_id,'cancelled'); return
                state = self.output.status(job).get('state')
                if state == 'queued' and self.output.system_status().get('current'):
                    self.output.cancel(job); self.inbox.finish(request_id,'busy'); return
                if state in {'completed','failed','cancelled','interrupted','expired'}:
                    self.inbox.finish(request_id, 'completed' if state=='completed' else 'cancelled' if state in {'cancelled','interrupted'} else state)
                    return
                self.sleep(.25)
            self.output.cancel(job); self.inbox.finish(request_id,'unknown')
        except Exception:
            if job:
                try: self.output.cancel(job)
                except Exception: pass
            self.inbox.finish(request_id,'unknown')
        finally:
            try: self.display.hug_idle(trace)
            except Exception: pass
