"""Owner-only Unix-socket API for the encrypted Face Vault."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from .client import DEFAULT_SOCKET
from .store import FaceVaultError, FaceVaultStore

MAX_REQUEST = 256 * 1024
OPERATIONS = {
    "status": "status", "identity.create": "identity_create", "identity.list": "identity_list",
    "identity.get": "identity_get", "template.add": "template_add", "template.delete": "template_delete",
    "identity.revoke": "identity_revoke", "match": "match",
    "identity.delete": "identity_delete", "vault.reset": "vault_reset", "audit.list": "audit_list",
}
RAW_MEDIA_KEYS = {"image", "image_path", "frame", "jpeg", "png", "video", "media", "media_id", "blob", "audio"}


def dispatch(store, request):
    if not isinstance(request, dict):
        raise FaceVaultError("invalid_request", "request must be an object")
    operation = request.get("operation")
    if operation not in OPERATIONS:
        raise FaceVaultError("unknown_operation", "unknown operation")
    arguments = {key: value for key, value in request.items() if key != "operation"}
    if RAW_MEDIA_KEYS.intersection(arguments):
        raise FaceVaultError("raw_media_rejected", "Face Vault accepts templates and metadata, never raw media")
    return getattr(store, OPERATIONS[operation])(**arguments)


async def handle(store, reader, writer):
    try:
        raw = await reader.readline()
        if not raw or len(raw) > MAX_REQUEST or not raw.endswith(b"\n"):
            raise FaceVaultError("invalid_request", "invalid request framing")
        result = await asyncio.to_thread(dispatch, store, json.loads(raw))
        response = {"ok": True, "result": result}
    except FaceVaultError as exc:
        response = {"ok": False, "code": exc.code, "error": str(exc)}
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        response = {"ok": False, "code": "invalid_arguments", "error": type(exc).__name__}
    except Exception as exc:
        response = {"ok": False, "code": "internal_error", "error": type(exc).__name__}
    writer.write((json.dumps(response, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
    await writer.drain(); writer.close(); await writer.wait_closed()


async def run(socket_path, database_path=None, key_path=None):
    path = Path(socket_path); path.parent.mkdir(parents=True, exist_ok=True); path.unlink(missing_ok=True)
    store = FaceVaultStore(database_path, key_path)
    server = await asyncio.start_unix_server(lambda r, w: handle(store, r, w), path=socket_path)
    os.chmod(path, 0o600)
    try:
        async with server: await server.serve_forever()
    finally:
        path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--socket", default=DEFAULT_SOCKET)
    parser.add_argument("--database"); parser.add_argument("--key"); args = parser.parse_args()
    asyncio.run(run(args.socket, args.database, args.key))


if __name__ == "__main__": main()
