"""Dependency-free client for the private Feature Store service."""

from __future__ import annotations

import json
import os
import socket


DEFAULT_SOCKET = os.environ.get("VQ_FEATURE_SOCKET", f"/run/user/{os.getuid()}/vq-features.sock")


class FeatureStoreClientError(RuntimeError):
    def __init__(self, code, message): super().__init__(message); self.code = code


class FeatureStoreClient:
    def __init__(self, socket_path=DEFAULT_SOCKET, *, timeout=5):
        self.socket_path, self.timeout = socket_path, timeout
    def call(self, operation, **values):
        body=json.dumps({"operation":operation,**values},ensure_ascii=False,separators=(",", ":"))+"\n"
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout); client.connect(self.socket_path); client.sendall(body.encode())
            raw=client.makefile("rb").readline(1024*1024)
        try: result=json.loads(raw)
        except (json.JSONDecodeError,UnicodeDecodeError) as exc:
            raise FeatureStoreClientError("invalid_response","invalid Feature Store response") from exc
        if not result.get("ok"):
            raise FeatureStoreClientError(result.get("code","service_error"),result.get("error","Feature Store error"))
        return result["result"]
    def schemas(self): return self.call("schemas.list")
    def write(self,owner,schema_name,values,source,**options):
        return self.call("feature.write",owner=owner,schema_name=schema_name,values=values,source=source,**options)
    def label(self,owner,feature_id,label,label_source="explicit-user",confidence=1.0):
        return self.call("feature.label",owner=owner,feature_id=feature_id,label=label,
                         label_source=label_source,confidence=confidence)
    def query(self,owner,schema_name,**options):
        return self.call("feature.query",owner=owner,schema_name=schema_name,**options)
    def snapshot(self,owner,schema_name,dataset_id,**options):
        return self.call("dataset.snapshot",owner=owner,schema_name=schema_name,dataset_id=dataset_id,**options)
    def get_snapshot(self,owner,snapshot_id):
        return self.call("dataset.get",owner=owner,snapshot_id=snapshot_id)
    def delete(self,owner,feature_id): return self.call("feature.delete",owner=owner,feature_id=feature_id)
    def stats(self): return self.call("stats")
