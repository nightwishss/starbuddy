"""SQLite audit trail for feedback, evaluation, canary and rollback."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path


IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
DIGEST = re.compile(r"^[a-f0-9]{64}$")
PRIVACY = {"local-private", "sensitive"}


class ImprovementError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class ImprovementStore:
    def __init__(self, path, *, clock=time.time):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        self.connection = sqlite3.connect(str(self.path), check_same_thread=False)
        os.chmod(self.path, 0o600)
        self.connection.row_factory = sqlite3.Row
        self.clock = clock
        self.lock = threading.Lock()
        self.connection.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS consent(
          scope TEXT PRIMARY KEY, allowed INTEGER NOT NULL, updated_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS feedback(
          feedback_id TEXT PRIMARY KEY, task TEXT NOT NULL, model_id TEXT NOT NULL,
          trace_id TEXT, rating INTEGER NOT NULL, tags_json TEXT NOT NULL,
          note TEXT, created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS datasets(
          dataset_id TEXT PRIMARY KEY, task TEXT NOT NULL, item_count INTEGER NOT NULL,
          schema_name TEXT NOT NULL, content_sha256 TEXT NOT NULL,
          privacy TEXT NOT NULL, created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS evaluations(
          run_id TEXT PRIMARY KEY, task TEXT NOT NULL, model_id TEXT NOT NULL,
          candidate_version TEXT NOT NULL, dataset_id TEXT NOT NULL,
          metrics_json TEXT NOT NULL, thresholds_json TEXT NOT NULL,
          passed INTEGER NOT NULL, created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS deployments(
          task TEXT PRIMARY KEY, active_model TEXT NOT NULL, previous_model TEXT,
          mode TEXT NOT NULL, canary_percent INTEGER NOT NULL, run_id TEXT,
          updated_at REAL NOT NULL);
        """)
        self.connection.commit()
        self._secure_files()

    def _secure_files(self):
        for path in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
            if path.exists():
                os.chmod(path, 0o600)

    def close(self):
        self.connection.close()

    @staticmethod
    def _id(value, field):
        if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
            raise ImprovementError("invalid_identifier", f"invalid {field}")
        return value

    def set_consent(self, *, scope, allowed):
        self._id(scope, "scope")
        if not isinstance(allowed, bool):
            raise ImprovementError("invalid_consent", "allowed must be boolean")
        now = self.clock()
        with self.lock:
            self.connection.execute(
                "INSERT INTO consent(scope,allowed,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(scope) DO UPDATE SET allowed=excluded.allowed,updated_at=excluded.updated_at",
                (scope, int(allowed), now))
            self.connection.commit()
        return {"scope": scope, "allowed": allowed, "updated_at": now}

    def consent_status(self, *, scope):
        self._id(scope, "scope")
        row = self.connection.execute("SELECT * FROM consent WHERE scope=?", (scope,)).fetchone()
        return {"scope": scope, "allowed": bool(row["allowed"]) if row else False,
                "updated_at": row["updated_at"] if row else None}

    def _require_consent(self, scope):
        if not self.consent_status(scope=scope)["allowed"]:
            raise ImprovementError("consent_required", f"explicit consent required for {scope}")

    def add_feedback(self, *, task, model_id, rating, tags=None, note=None, trace_id=None):
        self._require_consent("improvement.feedback")
        self._id(task, "task"); self._id(model_id, "model_id")
        if trace_id is not None: self._id(trace_id, "trace_id")
        if rating not in {-1, 0, 1} or isinstance(rating, bool):
            raise ImprovementError("invalid_rating", "rating must be -1, 0 or 1")
        if tags is None: tags = []
        if not isinstance(tags, list) or len(tags) > 12 or any(not isinstance(x, str) or len(x) > 32 for x in tags):
            raise ImprovementError("invalid_tags", "invalid feedback tags")
        if note is not None and (not isinstance(note, str) or len(note) > 500):
            raise ImprovementError("invalid_note", "feedback note too long")
        # Feedback is semantic metadata only. Media bytes, paths and base64 are not accepted fields.
        feedback_id = "feedback_" + uuid.uuid4().hex
        now = self.clock()
        with self.lock:
            self.connection.execute("INSERT INTO feedback VALUES(?,?,?,?,?,?,?,?)", (
                feedback_id, task, model_id, trace_id, rating,
                json.dumps(tags, separators=(",", ":")), note, now))
            self.connection.commit()
        return {"feedback_id": feedback_id, "created_at": now, "raw_media_stored": False}

    def register_dataset(self, *, dataset_id, task, item_count, schema_name,
                         content_sha256, privacy="local-private"):
        self._require_consent("improvement.datasets")
        self._id(dataset_id, "dataset_id"); self._id(task, "task"); self._id(schema_name, "schema")
        if not isinstance(item_count, int) or item_count < 1:
            raise ImprovementError("invalid_count", "item_count must be positive")
        if not isinstance(content_sha256, str) or not DIGEST.fullmatch(content_sha256):
            raise ImprovementError("invalid_digest", "content digest must be SHA-256")
        if privacy not in PRIVACY:
            raise ImprovementError("invalid_privacy", "invalid privacy label")
        now = self.clock()
        with self.lock:
            existing = self.connection.execute(
                "SELECT task,item_count,schema_name,content_sha256,privacy,created_at "
                "FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
            if existing is not None:
                expected = (task, item_count, schema_name, content_sha256, privacy)
                actual = tuple(existing[key] for key in
                               ("task", "item_count", "schema_name", "content_sha256", "privacy"))
                if actual != expected:
                    raise ImprovementError("dataset_conflict", "dataset id already has different content")
                return {"dataset_id": dataset_id, "registered": True, "existing": True,
                        "created_at": existing["created_at"], "raw_media_stored": False}
            self.connection.execute("INSERT INTO datasets VALUES(?,?,?,?,?,?,?)", (
                dataset_id, task, item_count, schema_name, content_sha256, privacy, now))
            self.connection.commit()
        return {"dataset_id": dataset_id, "registered": True, "existing": False,
                "created_at": now, "raw_media_stored": False}

    def record_evaluation(self, *, task, model_id, candidate_version, dataset_id,
                          metrics, thresholds):
        for value, field in ((task, "task"), (model_id, "model_id"),
                             (candidate_version, "candidate_version"), (dataset_id, "dataset_id")):
            self._id(value, field)
        if not isinstance(metrics, dict) or not isinstance(thresholds, dict) or not thresholds:
            raise ImprovementError("invalid_metrics", "metrics and thresholds must be objects")
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in metrics.values()):
            raise ImprovementError("invalid_metrics", "metric values must be numeric")
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in thresholds.values()):
            raise ImprovementError("invalid_metrics", "threshold values must be numeric")
        dataset = self.connection.execute("SELECT task FROM datasets WHERE dataset_id=?", (dataset_id,)).fetchone()
        if dataset is None or dataset["task"] != task:
            raise ImprovementError("dataset_not_found", "matching dataset manifest not found")
        passed = all(key in metrics and metrics[key] >= value for key, value in thresholds.items())
        run_id = "eval_" + uuid.uuid4().hex
        now = self.clock()
        with self.lock:
            self.connection.execute("INSERT INTO evaluations VALUES(?,?,?,?,?,?,?,?,?)", (
                run_id, task, model_id, candidate_version, dataset_id,
                json.dumps(metrics, separators=(",", ":")),
                json.dumps(thresholds, separators=(",", ":")), int(passed), now))
            self.connection.commit()
        return {"run_id": run_id, "passed": passed, "created_at": now}

    def evaluation(self, *, run_id):
        self._id(run_id, "run_id")
        row = self.connection.execute("SELECT * FROM evaluations WHERE run_id=?", (run_id,)).fetchone()
        if row is None: raise ImprovementError("evaluation_not_found", "evaluation not found")
        value = dict(row); value["passed"] = bool(value["passed"])
        value["metrics"] = json.loads(value.pop("metrics_json"))
        value["thresholds"] = json.loads(value.pop("thresholds_json"))
        return value

    def seed_deployment(self, *, task, model_id):
        self._id(task, "task"); self._id(model_id, "model_id")
        now = self.clock()
        with self.lock:
            row = self.connection.execute("SELECT * FROM deployments WHERE task=?", (task,)).fetchone()
            if row is not None:
                if row["active_model"] != model_id:
                    raise ImprovementError("deployment_exists", "task already has another deployment baseline")
                return self.deployment(task=task)
            self.connection.execute("INSERT INTO deployments VALUES(?,?,?,?,?,?,?)",
                                    (task, model_id, None, "production", 0, None, now))
            self.connection.commit()
        return self.deployment(task=task)

    def promote(self, *, task, model_id, run_id, canary_percent=10):
        self._id(task, "task"); self._id(model_id, "model_id"); self._id(run_id, "run_id")
        if not isinstance(canary_percent, int) or not 1 <= canary_percent <= 25:
            raise ImprovementError("invalid_canary", "canary must be 1..25 percent")
        row = self.connection.execute(
            "SELECT * FROM evaluations WHERE run_id=? AND task=? AND model_id=? AND passed=1",
            (run_id, task, model_id)).fetchone()
        if row is None:
            raise ImprovementError("evaluation_required", "passing evaluation required")
        current = self.connection.execute("SELECT active_model FROM deployments WHERE task=?", (task,)).fetchone()
        previous = current["active_model"] if current else None
        now = self.clock()
        with self.lock:
            self.connection.execute(
                "INSERT INTO deployments VALUES(?,?,?,?,?,?,?) ON CONFLICT(task) DO UPDATE SET "
                "active_model=excluded.active_model,previous_model=excluded.previous_model,mode=excluded.mode,"
                "canary_percent=excluded.canary_percent,run_id=excluded.run_id,updated_at=excluded.updated_at",
                (task, model_id, previous, "canary", canary_percent, run_id, now))
            self.connection.commit()
        return self.deployment(task=task)

    def rollback(self, *, task):
        self._id(task, "task")
        row = self.connection.execute("SELECT * FROM deployments WHERE task=?", (task,)).fetchone()
        if row is None or not row["previous_model"]:
            raise ImprovementError("rollback_unavailable", "no previous model")
        now = self.clock()
        with self.lock:
            self.connection.execute(
                "UPDATE deployments SET active_model=?,previous_model=?,mode='rolled_back',"
                "canary_percent=0,updated_at=? WHERE task=?",
                (row["previous_model"], row["active_model"], now, task))
            self.connection.commit()
        return self.deployment(task=task)

    def deployment(self, *, task):
        self._id(task, "task")
        row = self.connection.execute("SELECT * FROM deployments WHERE task=?", (task,)).fetchone()
        return dict(row) if row else {"task": task, "mode": "unmanaged", "active_model": None}

    def stats(self):
        result = {"raw_media_stored": False, "automatic_training": False,
                  "automatic_promotion": False, "database": "sqlite-local-0600"}
        for table in ("feedback", "datasets", "evaluations", "deployments"):
            result[table] = self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return result
