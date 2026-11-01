"""Internal synchronous client for Face Vault; not an Agent capability."""

import json
import os
import socket

DEFAULT_SOCKET = os.environ.get("VQ_FACE_VAULT_SOCKET", "/run/user/1000/vq-face-vault.sock")
MAX_RESPONSE = 256 * 1024


class FaceVaultClientError(RuntimeError):
    pass


class FaceVaultClient:
    def __init__(self, socket_path=DEFAULT_SOCKET, timeout=3.0):
        self.socket_path, self.timeout = socket_path, timeout

    def call(self, operation, **values):
        payload = (json.dumps({"operation": operation, **values}, ensure_ascii=False) + "\n").encode()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout); client.connect(self.socket_path); client.sendall(payload)
            response = client.makefile("rb").readline(MAX_RESPONSE + 1)
        if not response or len(response) > MAX_RESPONSE:
            raise FaceVaultClientError("invalid Face Vault response")
        value = json.loads(response)
        if not value.get("ok"):
            raise FaceVaultClientError(f"{value.get('code')}: {value.get('error')}")
        return value["result"]

    def status(self): return self.call("status")
    def identities(self): return self.call("identity.list")
    def identity(self, identity_id): return self.call("identity.get", identity_id=identity_id)
    def create_identity(self, *, consent, consent_version, profile_id=None, label=None):
        return self.call("identity.create", consent=consent, consent_version=consent_version,
                         profile_id=profile_id, label=label)
    def add_template(self, identity_id, model_id, embedding, *, consent, quality=None):
        return self.call("template.add", identity_id=identity_id, model_id=model_id,
                         embedding=embedding, consent=consent, quality=quality)
    def match(self, model_id, embedding):
        return self.call("match", model_id=model_id, embedding=embedding)
    def delete_identity(self, identity_id, *, consent):
        return self.call("identity.delete", identity_id=identity_id, consent=consent)
