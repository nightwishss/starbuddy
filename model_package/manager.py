"""Create and verify HMAC-authenticated local model packages."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
from pathlib import Path


IDENTIFIER=re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
DIGEST=re.compile(r"^[a-f0-9]{64}$")
MAX_CANDIDATE_BYTES=1024*1024
MAX_PACKAGE_BYTES=2*1024*1024


class PackageError(ValueError):
    def __init__(self,code,message): super().__init__(message); self.code=code


class PackageManager:
    def __init__(self,candidate_root,package_root,key_path,improvement):
        self.candidate_root=Path(candidate_root); self.package_root=Path(package_root)
        self.key_path=Path(key_path); self.improvement=improvement

    @staticmethod
    def _id(value,field):
        if not isinstance(value,str) or not IDENTIFIER.fullmatch(value):
            raise PackageError("invalid_identifier",f"invalid {field}")

    def _key(self):
        self.key_path.parent.mkdir(parents=True,exist_ok=True); os.chmod(self.key_path.parent,0o700)
        if not self.key_path.exists():
            try:
                descriptor=os.open(self.key_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
                with os.fdopen(descriptor,"wb") as output: output.write(secrets.token_bytes(32))
            except FileExistsError: pass
        if self.key_path.stat().st_mode&0o777 != 0o600:
            raise PackageError("insecure_key","package key permissions must be 0600")
        key=self.key_path.read_bytes()
        if len(key)!=32: raise PackageError("invalid_key","package key must be 32 bytes")
        return key

    def package(self,*,model_id,candidate_version,expected_sha256):
        self._id(model_id,"model_id"); self._id(candidate_version,"candidate_version")
        if not isinstance(expected_sha256,str) or not DIGEST.fullmatch(expected_sha256):
            raise PackageError("invalid_digest","expected SHA-256 required")
        candidate_path=self.candidate_root/model_id/f"{candidate_version}.json"
        try: body=candidate_path.read_bytes()
        except OSError as exc: raise PackageError("candidate_not_found","candidate not found") from exc
        if len(body)>MAX_CANDIDATE_BYTES: raise PackageError("candidate_too_large","candidate exceeds 1 MiB")
        if hashlib.sha256(body).hexdigest()!=expected_sha256:
            raise PackageError("candidate_integrity","candidate digest mismatch")
        try: candidate=json.loads(body)
        except (json.JSONDecodeError,UnicodeDecodeError) as exc:
            raise PackageError("invalid_candidate","candidate must be UTF-8 JSON") from exc
        required={"format":"vq-linear-classifier-v1","task":"posture.classifier",
                  "model_id":model_id,"candidate_version":candidate_version,"deployable":False}
        if not isinstance(candidate,dict) or any(candidate.get(key)!=value for key,value in required.items()):
            raise PackageError("invalid_candidate","candidate identity or format mismatch")
        evaluation=candidate.get("evaluation") or {}; run_id=evaluation.get("run_id")
        if not evaluation.get("passed") or not isinstance(run_id,str):
            raise PackageError("evaluation_required","passing embedded evaluation required")
        authoritative=self.improvement.call("evaluation.get",run_id=run_id)
        expected=("posture.classifier",model_id,candidate_version,True)
        actual=(authoritative.get("task"),authoritative.get("model_id"),
                authoritative.get("candidate_version"),authoritative.get("passed"))
        if actual!=expected: raise PackageError("evaluation_mismatch","authoritative evaluation mismatch")
        payload={"package_format":"vq-model-package-v1","task":"posture.classifier",
                 "model_id":model_id,"candidate_version":candidate_version,"candidate_sha256":expected_sha256,
                 "evaluation_run_id":run_id,"candidate":candidate}
        canonical=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",", ":")).encode()
        signature=hmac.new(self._key(),canonical,hashlib.sha256).hexdigest()
        package={"payload":payload,"signature":{"algorithm":"hmac-sha256-local","value":signature}}
        package_body=(json.dumps(package,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()
        directory=self.package_root/model_id; directory.mkdir(parents=True,exist_ok=True)
        os.chmod(self.package_root,0o700); os.chmod(directory,0o700)
        path=directory/f"{candidate_version}.vqmodel"
        try: descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        except FileExistsError as exc: raise PackageError("package_exists","package version already exists") from exc
        try:
            with os.fdopen(descriptor,"wb") as output: output.write(package_body)
        except Exception:
            path.unlink(missing_ok=True); raise
        return {"package_path":str(path),"package_sha256":hashlib.sha256(package_body).hexdigest(),
                "signature":signature,"evaluation_run_id":run_id,"runtime_applied":False}

    def verify(self,path):
        path=Path(path)
        try:
            body=path.read_bytes()
            if len(body)>MAX_PACKAGE_BYTES: raise PackageError("package_too_large","package exceeds 2 MiB")
            package=json.loads(body)
        except (OSError,json.JSONDecodeError,UnicodeDecodeError) as exc:
            raise PackageError("invalid_package","cannot read package") from exc
        payload=package.get("payload") if isinstance(package,dict) else None
        signature=(package.get("signature") or {}).get("value") if isinstance(package,dict) else None
        if not isinstance(payload,dict) or not isinstance(signature,str):
            raise PackageError("invalid_package","invalid package envelope")
        canonical=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(",", ":")).encode()
        expected=hmac.new(self._key(),canonical,hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature,expected): raise PackageError("invalid_signature","package signature mismatch")
        candidate_body=(json.dumps(payload.get("candidate"),ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()
        if hashlib.sha256(candidate_body).hexdigest()!=payload.get("candidate_sha256"):
            raise PackageError("candidate_integrity","packaged candidate digest mismatch")
        return {"verified":True,"payload":payload,"runtime_applied":False}
