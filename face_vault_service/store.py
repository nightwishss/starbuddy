"""Encrypted local store for explicitly enrolled face templates."""

from __future__ import annotations

import math
import os
import re
import sqlite3
import struct
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .crypto import AuthenticatedCipher, NONCE_BYTES, load_or_create_key


CONSENTS = {"user_confirmed", "guardian_confirmed"}
ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
MODEL_RE = re.compile(r"^[A-Za-z0-9_.:/+-]{1,128}$")
RESET_CONFIRMATION = "DELETE_ALL_BIOMETRICS"
SCHEMA_VERSION = 1
MAX_IDENTITIES = 32
MAX_TEMPLATES_PER_IDENTITY = 8
MATCH_THRESHOLD = 0.42
AMBIGUITY_MARGIN = 0.08


class FaceVaultError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def default_database_path() -> Path:
    return Path(os.environ.get("VQ_FACE_VAULT_DB", "~/.local/share/vq-platform/face-vault/face-vault.sqlite3")).expanduser()


def default_key_path() -> Path:
    return Path(os.environ.get("VQ_FACE_VAULT_KEY", "~/.config/vq-platform/secrets/face-vault.key")).expanduser()


class FaceVaultStore:
    """Metadata API plus encrypted template writes; matching is deliberately absent in M3."""

    def __init__(self, path: str | Path | None = None, key_path: str | Path | None = None, *, clock=time.time):
        self.path = Path(path or default_database_path())
        self.key_path = Path(key_path or default_key_path())
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        self._cipher = AuthenticatedCipher(load_or_create_key(self.key_path))
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        os.chmod(self.path, 0o600)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA secure_delete=ON")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS identities (
              identity_id TEXT PRIMARY KEY, profile_id TEXT, label TEXT,
              status TEXT NOT NULL CHECK(status IN ('active','revoked')),
              consent TEXT NOT NULL, consent_version TEXT NOT NULL,
              consent_granted_at REAL NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS templates (
              template_id TEXT PRIMARY KEY, identity_id TEXT NOT NULL REFERENCES identities(identity_id) ON DELETE CASCADE,
              model_id TEXT NOT NULL, dimension INTEGER NOT NULL, nonce BLOB NOT NULL,
              ciphertext BLOB NOT NULL, quality REAL, created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS templates_identity ON templates(identity_id, created_at);
            CREATE TABLE IF NOT EXISTS audit (
              event_id INTEGER PRIMARY KEY AUTOINCREMENT, operation TEXT NOT NULL,
              identity_id TEXT, template_id TEXT, result TEXT NOT NULL, occurred_at REAL NOT NULL
            );
        """)
        self._db.commit()

    @staticmethod
    def _consent(value: Any) -> str:
        if value not in CONSENTS:
            raise FaceVaultError("consent_required", "explicit user or guardian confirmation is required")
        return value

    @staticmethod
    def _identifier(value: Any, *, field="identity_id") -> str:
        if not isinstance(value, str) or not ID_RE.fullmatch(value):
            raise FaceVaultError(f"invalid_{field}", f"invalid {field}")
        return value

    @staticmethod
    def _optional_text(value: Any, field: str, limit: int) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit:
            raise FaceVaultError(f"invalid_{field}", f"invalid {field}")
        return value.strip()

    def _audit(self, operation: str, result: str, identity_id=None, template_id=None) -> None:
        self._db.execute("INSERT INTO audit(operation,identity_id,template_id,result,occurred_at) VALUES(?,?,?,?,?)",
                         (operation, identity_id, template_id, result, self.clock()))

    def status(self) -> dict:
        with self._lock:
            identities = self._db.execute("SELECT COUNT(*) FROM identities WHERE status='active'").fetchone()[0]
            revoked = self._db.execute("SELECT COUNT(*) FROM identities WHERE status='revoked'").fetchone()[0]
            templates = self._db.execute("SELECT COUNT(*) FROM templates").fetchone()[0]
        return {"active_identities": identities, "revoked_identities": revoked, "templates": templates,
                "encrypted_at_rest": True, "encryption": "chacha20-poly1305", "raw_images_allowed": False,
                "matching_enabled": True, "agent_exposed": False, "schema_version": SCHEMA_VERSION,
                "max_identities": MAX_IDENTITIES, "max_templates_per_identity": MAX_TEMPLATES_PER_IDENTITY,
                "match_threshold": MATCH_THRESHOLD, "ambiguity_margin": AMBIGUITY_MARGIN}

    def identity_create(self, *, consent: str, consent_version: str, profile_id=None, label=None) -> dict:
        consent = self._consent(consent)
        consent_version = self._optional_text(consent_version, "consent_version", 64)
        profile_id = None if profile_id is None else self._identifier(profile_id, field="profile_id")
        label = self._optional_text(label, "label", 96)
        now, identity_id = self.clock(), "identity-" + uuid.uuid4().hex
        with self._lock, self._db:
            count = self._db.execute("SELECT COUNT(*) FROM identities WHERE status='active'").fetchone()[0]
            if count >= MAX_IDENTITIES:
                raise FaceVaultError("identity_limit", "active identity limit reached")
            self._db.execute("INSERT INTO identities VALUES(?,?,?,?,?,?,?,?,?)",
                             (identity_id, profile_id, label, "active", consent, consent_version, now, now, now))
            self._audit("identity.create", "created", identity_id)
        return self.identity_get(identity_id)

    def identity_list(self) -> dict:
        with self._lock:
            rows = self._db.execute("SELECT identity_id,profile_id,label,status,consent,consent_version,consent_granted_at,created_at,updated_at FROM identities ORDER BY created_at").fetchall()
        return {"items": [dict(row) for row in rows]}

    def identity_get(self, identity_id: str) -> dict:
        identity_id = self._identifier(identity_id)
        with self._lock:
            row = self._db.execute("SELECT * FROM identities WHERE identity_id=?", (identity_id,)).fetchone()
            if not row:
                raise FaceVaultError("identity_not_found", "identity does not exist")
            templates = self._db.execute("SELECT template_id,model_id,dimension,quality,created_at FROM templates WHERE identity_id=? ORDER BY created_at", (identity_id,)).fetchall()
        result = {key: row[key] for key in ("identity_id", "profile_id", "label", "status", "consent", "consent_version", "consent_granted_at", "created_at", "updated_at")}
        result["templates"] = [dict(item) for item in templates]
        return result

    @staticmethod
    def _embedding(value: Any) -> list[float]:
        if not isinstance(value, list) or not 32 <= len(value) <= 2048:
            raise FaceVaultError("invalid_embedding", "embedding dimension must be between 32 and 2048")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or abs(float(item)) > 100 for item in value):
            raise FaceVaultError("invalid_embedding", "embedding must contain finite bounded numbers")
        return [float(item) for item in value]

    @staticmethod
    def _aad(identity_id: str, template_id: str, model_id: str, dimension: int) -> bytes:
        return f"face-vault:v{SCHEMA_VERSION}:{identity_id}:{template_id}:{model_id}:{dimension}".encode()

    def template_add(self, identity_id: str, model_id: str, embedding: list[float], *, consent: str, quality=None) -> dict:
        identity_id, consent = self._identifier(identity_id), self._consent(consent)
        if not isinstance(model_id, str) or not MODEL_RE.fullmatch(model_id):
            raise FaceVaultError("invalid_model_id", "invalid model_id")
        vector = self._embedding(embedding)
        if quality is not None and (isinstance(quality, bool) or not isinstance(quality, (int, float)) or not 0 <= float(quality) <= 1):
            raise FaceVaultError("invalid_quality", "quality must be between 0 and 1")
        with self._lock, self._db:
            identity = self._db.execute("SELECT status FROM identities WHERE identity_id=?", (identity_id,)).fetchone()
            if not identity:
                raise FaceVaultError("identity_not_found", "identity does not exist")
            if identity["status"] != "active":
                raise FaceVaultError("identity_revoked", "templates cannot be added to a revoked identity")
            count = self._db.execute("SELECT COUNT(*) FROM templates WHERE identity_id=?", (identity_id,)).fetchone()[0]
            if count >= MAX_TEMPLATES_PER_IDENTITY:
                raise FaceVaultError("template_limit", "template limit reached for identity")
            template_id, now = "template-" + uuid.uuid4().hex, self.clock()
            plaintext = struct.pack(f"<{len(vector)}f", *vector)
            nonce = os.urandom(NONCE_BYTES)
            ciphertext = self._cipher.encrypt(nonce, plaintext, self._aad(identity_id, template_id, model_id, len(vector)))
            self._db.execute("INSERT INTO templates VALUES(?,?,?,?,?,?,?,?)",
                             (template_id, identity_id, model_id, len(vector), nonce, ciphertext,
                              None if quality is None else float(quality), now))
            self._audit("template.add", "created", identity_id, template_id)
        return {"template_id": template_id, "identity_id": identity_id, "model_id": model_id,
                "dimension": len(vector), "quality": None if quality is None else float(quality),
                "consent": consent, "created_at": now}

    def _decrypt_template_for_matching(self, template_id: str) -> list[float]:
        """Private future matcher hook; never exposed by the socket service."""
        template_id = self._identifier(template_id, field="template_id")
        with self._lock:
            row = self._db.execute("SELECT * FROM templates WHERE template_id=?", (template_id,)).fetchone()
        if not row:
            raise FaceVaultError("template_not_found", "template does not exist")
        plaintext = self._cipher.decrypt(row["nonce"], row["ciphertext"], self._aad(row["identity_id"], row["template_id"], row["model_id"], row["dimension"]))
        return list(struct.unpack(f"<{row['dimension']}f", plaintext))

    def match(self, model_id: str, embedding: list[float]) -> dict:
        """Match inside the custody boundary; never return stored or probe vectors."""
        if not isinstance(model_id, str) or not MODEL_RE.fullmatch(model_id):
            raise FaceVaultError("invalid_model_id", "invalid model_id")
        probe = self._embedding(embedding)
        probe_norm = math.sqrt(math.fsum(value * value for value in probe))
        if probe_norm <= 1e-12:
            raise FaceVaultError("invalid_embedding", "zero-length embedding cannot be matched")
        with self._lock:
            rows = self._db.execute(
                "SELECT t.*,i.profile_id,i.label FROM templates t JOIN identities i ON i.identity_id=t.identity_id "
                "WHERE i.status='active' AND t.model_id=? AND t.dimension=?",
                (model_id, len(probe)),
            ).fetchall()
            scores = {}
            metadata = {}
            for row in rows:
                stored = self._cipher.decrypt(
                    row["nonce"], row["ciphertext"],
                    self._aad(row["identity_id"], row["template_id"], row["model_id"], row["dimension"]),
                )
                vector = struct.unpack(f"<{row['dimension']}f", stored)
                norm = math.sqrt(math.fsum(value * value for value in vector))
                if norm <= 1e-12:
                    continue
                similarity = math.fsum(left * right for left, right in zip(probe, vector)) / (probe_norm * norm)
                scores[row["identity_id"]] = max(similarity, scores.get(row["identity_id"], -1.0))
                metadata[row["identity_id"]] = (row["profile_id"], row["label"])
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            return {"status": "unknown", "reason": "no_compatible_templates", "similarity": None,
                    "threshold": MATCH_THRESHOLD}
        identity_id, best = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else None
        if best < MATCH_THRESHOLD:
            return {"status": "unknown", "reason": "below_threshold", "similarity": round(best, 4),
                    "threshold": MATCH_THRESHOLD}
        if runner_up is not None and best - runner_up < AMBIGUITY_MARGIN:
            return {"status": "unknown", "reason": "ambiguous", "similarity": round(best, 4),
                    "threshold": MATCH_THRESHOLD}
        profile_id, label = metadata[identity_id]
        return {"status": "known", "identity_id": identity_id, "profile_id": profile_id, "label": label,
                "similarity": round(best, 4), "threshold": MATCH_THRESHOLD}

    def identity_revoke(self, identity_id: str, *, consent: str) -> dict:
        identity_id, consent = self._identifier(identity_id), self._consent(consent)
        now = self.clock()
        with self._lock, self._db:
            row = self._db.execute("SELECT status FROM identities WHERE identity_id=?", (identity_id,)).fetchone()
            if not row:
                raise FaceVaultError("identity_not_found", "identity does not exist")
            removed = self._db.execute("DELETE FROM templates WHERE identity_id=?", (identity_id,)).rowcount
            self._db.execute("UPDATE identities SET status='revoked',updated_at=? WHERE identity_id=?", (now, identity_id))
            self._audit("identity.revoke", "revoked", identity_id)
        return {"identity_id": identity_id, "revoked": True, "removed_templates": removed, "consent": consent}

    def template_delete(self, template_id: str, *, consent: str) -> dict:
        template_id, consent = self._identifier(template_id, field="template_id"), self._consent(consent)
        with self._lock, self._db:
            row = self._db.execute("SELECT identity_id FROM templates WHERE template_id=?", (template_id,)).fetchone()
            if not row:
                raise FaceVaultError("template_not_found", "template does not exist")
            identity_id = row["identity_id"]
            self._db.execute("DELETE FROM templates WHERE template_id=?", (template_id,))
            self._audit("template.delete", "deleted", identity_id, template_id)
        return {"template_id": template_id, "identity_id": identity_id, "deleted": True, "consent": consent}

    def identity_delete(self, identity_id: str, *, consent: str) -> dict:
        identity_id, consent = self._identifier(identity_id), self._consent(consent)
        with self._lock, self._db:
            row = self._db.execute("SELECT 1 FROM identities WHERE identity_id=?", (identity_id,)).fetchone()
            if not row:
                raise FaceVaultError("identity_not_found", "identity does not exist")
            removed = self._db.execute("SELECT COUNT(*) FROM templates WHERE identity_id=?", (identity_id,)).fetchone()[0]
            self._audit("identity.delete", "deleted", identity_id)
            self._db.execute("DELETE FROM identities WHERE identity_id=?", (identity_id,))
        return {"identity_id": identity_id, "deleted": True, "removed_templates": removed, "consent": consent}

    def vault_reset(self, *, consent: str, confirmation: str) -> dict:
        consent = self._consent(consent)
        if confirmation != RESET_CONFIRMATION:
            raise FaceVaultError("confirmation_required", f"confirmation must be {RESET_CONFIRMATION}")
        with self._lock, self._db:
            identities = self._db.execute("SELECT COUNT(*) FROM identities").fetchone()[0]
            templates = self._db.execute("SELECT COUNT(*) FROM templates").fetchone()[0]
            self._db.execute("DELETE FROM identities")
            self._audit("vault.reset", "reset")
        return {"reset": True, "removed_identities": identities, "removed_templates": templates, "consent": consent}

    def audit_list(self, *, limit=50) -> dict:
        if not isinstance(limit, int) or not 1 <= limit <= 200:
            raise FaceVaultError("invalid_limit", "audit limit must be between 1 and 200")
        with self._lock:
            rows = self._db.execute("SELECT event_id,operation,identity_id,template_id,result,occurred_at FROM audit ORDER BY event_id DESC LIMIT ?", (limit,)).fetchall()
        return {"items": [dict(row) for row in rows]}
