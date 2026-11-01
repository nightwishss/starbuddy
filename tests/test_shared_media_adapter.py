#!/usr/bin/env python3
"""Phone adapter tests for the shared media data plane."""

import tempfile
import unittest
from pathlib import Path

from gateway.media import MediaError, MediaUnavailableError, SharedMediaStoreAdapter
from media_service import MediaClientError


class FakeClient:
    def __init__(self, root: Path):
        self.root = root
        self.value = {
            "media_id": "audio_ABCDEFGHIJKLMNOPQRSTUVWXYZ", "kind": "audio",
            "mime_type": "audio/wav", "size": 16, "created_at": 1.0,
            "expires_at": 601.0,
        }
        self.path = root / "sample.wav"
        self.path.write_bytes(b"RIFFxxxxWAVEdata")

    def create_from_bytes(self, owner, mime_type, data, *, kind=None):
        self.created = (owner, mime_type, data, kind)
        return self.value

    def call(self, op, **arguments):
        self.last_call = (op, arguments)
        if op == "info":
            return self.value
        if op == "resolve":
            return {"media_id": self.value["media_id"], "path": str(self.path)}
        if op == "cleanup":
            return {"removed": 2}
        if op == "release":
            return {"media_id": arguments["media_id"], "released": True}
        raise AssertionError(op)


class SharedMediaAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.client = FakeClient(self.root)
        self.adapter = SharedMediaStoreAdapter(self.client, self.root / "work")

    def tearDown(self):
        self.temp.cleanup()

    def test_create_and_resolve_preserve_owner_boundary(self):
        item = self.adapter.create("phone-session", "audio/wav", b"RIFFxxxxWAVEdata")
        self.assertEqual(item.media_id, self.client.value["media_id"])
        self.assertNotIn("owner", item.public())
        resolved = self.adapter.get(item.media_id, "phone-session")
        self.assertEqual(resolved.path, self.client.path)
        self.assertEqual(self.adapter.consume(item.media_id, "phone-session"), b"RIFFxxxxWAVEdata")
        self.adapter.release(item.media_id, "phone-session")
        self.assertEqual(self.client.last_call[0], "release")

    def test_service_errors_remain_media_errors(self):
        self.client.call = lambda *args, **kwargs: (_ for _ in ()).throw(
            MediaClientError("PERMISSION_DENIED", "无权访问该媒体"))
        with self.assertRaisesRegex(MediaError, "无权"):
            self.adapter.get(self.client.value["media_id"], "wrong")

    def test_transport_failure_is_reported_as_unavailable(self):
        self.client.create_from_bytes = lambda *args, **kwargs: (_ for _ in ()).throw(
            ConnectionRefusedError())
        with self.assertRaises(MediaUnavailableError):
            self.adapter.create("phone-session", "audio/wav", b"RIFFxxxxWAVEdata")


if __name__ == "__main__":
    unittest.main()
