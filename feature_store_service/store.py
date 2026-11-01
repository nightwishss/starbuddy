"""Private SQLite feature records, explicit labels and immutable snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .schema import SchemaRegistry


IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
FEATURE_ID = re.compile(r"^feature_[a-f0-9]{32}$")
LABEL_SOURCES = {"explicit-user", "operator", "verified-rule"}


class FeatureError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class FeatureStore:
    def __init__(self, path, *, consent, schemas=None, clock=time.time,
                 max_records=50_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(self.path.parent, 0o700)
        self.consent = consent
        self.schemas = schemas or SchemaRegistry()
        self.clock = clock
        self.max_records = max_records
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        PRAGMA foreign_keys=ON;
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS features(
          feature_id TEXT PRIMARY KEY, owner TEXT NOT NULL, schema_name TEXT NOT NULL,
          subject_id TEXT, source TEXT NOT NULL, trace_id TEXT, values_json TEXT NOT NULL,
          created_at REAL NOT NULL, expires_at REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS features_query ON features(owner,schema_name,created_at);
        CREATE TABLE IF NOT EXISTS labels(
          feature_id TEXT PRIMARY KEY REFERENCES features(feature_id) ON DELETE CASCADE,
          label_json TEXT NOT NULL, label_source TEXT NOT NULL, confidence REAL NOT NULL,
          created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS snapshots(
          snapshot_id TEXT PRIMARY KEY, owner TEXT NOT NULL, dataset_id TEXT UNIQUE NOT NULL,
          schema_name TEXT NOT NULL, item_count INTEGER NOT NULL, labelled_count INTEGER NOT NULL,
          content_sha256 TEXT NOT NULL, created_at REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS snapshot_items(
          snapshot_id TEXT NOT NULL, ordinal INTEGER NOT NULL, record_json TEXT NOT NULL,
          FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id) ON DELETE CASCADE,
          PRIMARY KEY(snapshot_id,ordinal));
        """)
        self.db.commit()
        self._secure()

    def _secure(self):
        for path in (self.path, Path(str(self.path)+"-wal"), Path(str(self.path)+"-shm")):
            if path.exists(): os.chmod(path, 0o600)

    def close(self): self.db.close()

    @staticmethod
    def _id(value, field):
        if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
            raise FeatureError("invalid_identifier", f"invalid {field}")
        return value

    def _allowed(self, schema_name):
        scope = "feature_store." + schema_name
        try: allowed = bool(self.consent(scope))
        except Exception as exc: raise FeatureError("consent_unavailable", "consent service unavailable") from exc
        if not allowed: raise FeatureError("consent_required", f"explicit consent required for {scope}")

    def _purge(self):
        self.db.execute("DELETE FROM features WHERE expires_at<=?", (self.clock(),))

    def write(self, *, owner, schema_name, values, source, subject_id=None,
              trace_id=None, retention_days=30):
        self._id(owner,"owner"); self._id(schema_name,"schema_name"); self._id(source,"source")
        if subject_id is not None: self._id(subject_id,"subject_id")
        if trace_id is not None: self._id(trace_id,"trace_id")
        if not isinstance(retention_days,int) or not 1 <= retention_days <= 90:
            raise FeatureError("invalid_retention", "retention must be 1..90 days")
        self._allowed(schema_name)
        try: clean = self.schemas.validate(schema_name, values)
        except ValueError as exc: raise FeatureError("schema_validation", str(exc)) from exc
        now=self.clock(); feature_id="feature_"+uuid.uuid4().hex
        with self.lock:
            self._purge()
            count=self.db.execute("SELECT COUNT(*) FROM features").fetchone()[0]
            if count >= self.max_records: raise FeatureError("quota_exceeded", "feature quota reached")
            self.db.execute("INSERT INTO features VALUES(?,?,?,?,?,?,?,?,?)",(
                feature_id,owner,schema_name,subject_id,source,trace_id,
                json.dumps(clean,ensure_ascii=False,separators=(",", ":")),now,
                now+retention_days*86400))
            self.db.commit(); self._secure()
        return {"feature_id":feature_id,"schema_name":schema_name,"created_at":now,
                "expires_at":now+retention_days*86400,"raw_media_stored":False}

    def label(self, *, owner, feature_id, label, label_source="explicit-user", confidence=1.0):
        self._id(owner,"owner")
        if not isinstance(feature_id,str) or not FEATURE_ID.fullmatch(feature_id):
            raise FeatureError("invalid_feature_id","invalid feature id")
        if label_source not in LABEL_SOURCES: raise FeatureError("invalid_label_source","invalid label source")
        if isinstance(confidence,bool) or not isinstance(confidence,(int,float)) or not 0 <= confidence <= 1:
            raise FeatureError("invalid_confidence","confidence must be 0..1")
        if not isinstance(label,(str,int,float,bool)) or isinstance(label,str) and (not label or len(label)>64):
            raise FeatureError("invalid_label","label must be a short scalar")
        if isinstance(label,float) and not math.isfinite(label):
            raise FeatureError("invalid_label","numeric label must be finite")
        with self.lock:
            self._purge()
            row=self.db.execute("SELECT schema_name FROM features WHERE feature_id=? AND owner=?",(feature_id,owner)).fetchone()
            if row is None: raise FeatureError("feature_not_found","feature not found")
            self._allowed(row["schema_name"])
            now=self.clock()
            self.db.execute("INSERT INTO labels VALUES(?,?,?,?,?) ON CONFLICT(feature_id) DO UPDATE SET "
                            "label_json=excluded.label_json,label_source=excluded.label_source,"
                            "confidence=excluded.confidence,created_at=excluded.created_at",
                            (feature_id,json.dumps(label,ensure_ascii=False),label_source,float(confidence),now))
            self.db.commit(); self._secure()
        return {"feature_id":feature_id,"label":label,"label_source":label_source,"confidence":confidence}

    def query(self, *, owner, schema_name, labelled=None, limit=100):
        self._id(owner,"owner"); self._id(schema_name,"schema_name")
        if labelled not in {None,True,False}: raise FeatureError("invalid_filter","labelled must be boolean")
        if not isinstance(limit,int) or not 1 <= limit <= 500: raise FeatureError("invalid_limit","invalid limit")
        self._allowed(schema_name)
        join="LEFT JOIN labels l ON l.feature_id=f.feature_id"
        where="f.owner=? AND f.schema_name=? AND f.expires_at>?"
        if labelled is True: where += " AND l.feature_id IS NOT NULL"
        if labelled is False: where += " AND l.feature_id IS NULL"
        rows=self.db.execute(f"SELECT f.*,l.label_json,l.label_source,l.confidence label_confidence "
                             f"FROM features f {join} WHERE {where} ORDER BY f.created_at DESC LIMIT ?",
                             (owner,schema_name,self.clock(),limit)).fetchall()
        return {"features":[self._public(row) for row in rows],"raw_media_stored":False}

    @staticmethod
    def _public(row):
        value={key:row[key] for key in ("feature_id","schema_name","subject_id","source","trace_id","created_at","expires_at")}
        value["values"]=json.loads(row["values_json"])
        value["label"]=json.loads(row["label_json"]) if row["label_json"] is not None else None
        value["label_source"]=row["label_source"]
        value["label_confidence"]=row["label_confidence"]
        return value

    def snapshot(self, *, owner, schema_name, dataset_id, labelled_only=True):
        self._id(dataset_id,"dataset_id")
        values=self.query(owner=owner,schema_name=schema_name,
                          labelled=True if labelled_only else None,limit=500)
        records=sorted(values["features"],key=lambda item:item["feature_id"])
        if not records: raise FeatureError("empty_dataset","no matching features")
        canonical=json.dumps(records,ensure_ascii=False,sort_keys=True,separators=(",", ":")).encode()
        digest=hashlib.sha256(canonical).hexdigest(); snapshot_id="snapshot_"+uuid.uuid4().hex
        labelled_count=sum(item["label"] is not None for item in records); now=self.clock()
        with self.lock:
            self.db.execute("INSERT INTO snapshots VALUES(?,?,?,?,?,?,?,?)",(
                snapshot_id,owner,dataset_id,schema_name,len(records),labelled_count,digest,now))
            self.db.executemany("INSERT INTO snapshot_items VALUES(?,?,?)",[
                (snapshot_id,index,json.dumps(record,ensure_ascii=False,sort_keys=True,separators=(",", ":")))
                for index,record in enumerate(records)
            ])
            self.db.commit(); self._secure()
        return {"snapshot_id":snapshot_id,"dataset_id":dataset_id,"schema_name":schema_name,
                "task":self.schemas.task(schema_name),
                "item_count":len(records),"labelled_count":labelled_count,
                "content_sha256":digest,"privacy":"local-private","raw_media_stored":False}

    def get_snapshot(self, *, owner, snapshot_id):
        self._id(owner,"owner"); self._id(snapshot_id,"snapshot_id")
        row=self.db.execute("SELECT * FROM snapshots WHERE snapshot_id=? AND owner=?",(snapshot_id,owner)).fetchone()
        if row is None: raise FeatureError("snapshot_not_found","snapshot not found")
        self._allowed(row["schema_name"])
        items=self.db.execute("SELECT record_json FROM snapshot_items WHERE snapshot_id=? ORDER BY ordinal",
                              (snapshot_id,)).fetchall()
        return {**dict(row),"records":[json.loads(item["record_json"]) for item in items],
                "raw_media_stored":False}

    def drop_snapshot(self, snapshot_id):
        with self.lock:
            self.db.execute("DELETE FROM snapshot_items WHERE snapshot_id=?",(snapshot_id,))
            self.db.execute("DELETE FROM snapshots WHERE snapshot_id=?",(snapshot_id,))
            self.db.commit()

    def delete(self, *, owner, feature_id):
        self._id(owner,"owner")
        if not isinstance(feature_id,str) or not FEATURE_ID.fullmatch(feature_id):
            raise FeatureError("invalid_feature_id","invalid feature id")
        with self.lock:
            # Explicit cleanup also covers databases created before foreign keys were enabled.
            self.db.execute("DELETE FROM labels WHERE feature_id IN "
                            "(SELECT feature_id FROM features WHERE owner=? AND feature_id=?)",
                            (owner,feature_id))
            cursor=self.db.execute("DELETE FROM features WHERE owner=? AND feature_id=?",(owner,feature_id))
            self.db.commit()
        return {"feature_id":feature_id,"deleted":cursor.rowcount==1}

    def stats(self):
        with self.lock:
            self._purge(); self.db.commit()
            return {"features":self.db.execute("SELECT COUNT(*) FROM features").fetchone()[0],
                    "labelled":self.db.execute("SELECT COUNT(*) FROM labels").fetchone()[0],
                    "snapshots":self.db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0],
                    "max_records":self.max_records,"raw_media_stored":False,
                    "database":"sqlite-local-0600"}
