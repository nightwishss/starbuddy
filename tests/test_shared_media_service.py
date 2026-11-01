#!/usr/bin/env python3
"""Shared media ownership, quota, expiry and restart recovery tests."""

import tempfile
import unittest
from pathlib import Path

from media_service.store import MediaError, SharedMediaStore


class SharedMediaStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.store = SharedMediaStore(self.root, default_ttl=10, global_limit=1024, max_objects=4)

    def tearDown(self):
        self.temp.cleanup()

    def create(self, owner="session-a", mime="image/jpeg", data=b"\xff\xd8pixels\xff\xd9", **kwargs):
        upload = self.store.begin(owner, mime, **kwargs)
        Path(upload["upload_path"]).write_bytes(data)
        return self.store.commit(upload["upload_id"], owner)

    def test_commit_uses_opaque_id_and_private_file(self):
        item = self.create()
        self.assertTrue(item.media_id.startswith("image_"))
        path = self.store.resolve(item.media_id, "session-a")
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertNotIn("owner_hash", item.public())

    def test_cross_owner_access_and_commit_are_denied(self):
        upload = self.store.begin("session-a", "audio/ogg")
        Path(upload["upload_path"]).write_bytes(b"OggSpayload")
        with self.assertRaisesRegex(MediaError, "无权"):
            self.store.commit(upload["upload_id"], "session-b")
        item = self.store.commit(upload["upload_id"], "session-a")
        with self.assertRaisesRegex(MediaError, "无权"):
            self.store.get(item.media_id, "session-b")

    def test_invalid_signature_size_and_ttl_are_rejected(self):
        upload = self.store.begin("session-a", "image/jpeg")
        Path(upload["upload_path"]).write_bytes(b"not-jpeg")
        with self.assertRaisesRegex(MediaError, "签名"):
            self.store.commit(upload["upload_id"], "session-a")
        with self.assertRaises(MediaError):
            self.store.begin("session-a", "audio/wav", ttl_seconds=3601)

    def test_expiry_removes_data_and_metadata(self):
        item = self.create()
        path = self.store.resolve(item.media_id, "session-a")
        self.assertEqual(self.store.cleanup(item.expires_at + 1), 1)
        self.assertFalse(path.exists())
        with self.assertRaises(MediaError):
            self.store.get(item.media_id, "session-a")

    def test_restart_recovers_authorized_object(self):
        item = self.create()
        recovered = SharedMediaStore(self.root, default_ttl=10, global_limit=1024, max_objects=4)
        self.assertEqual(recovered.get(item.media_id, "session-a").size, item.size)
        with self.assertRaises(MediaError):
            recovered.get(item.media_id, "session-b")

    def test_orphans_and_partial_uploads_are_removed_on_restart(self):
        orphan = self.store.objects_root / "image_orphan.jpg"
        orphan.write_bytes(b"orphan")
        metadata_temp = self.store.objects_root / ".image_interrupted.deadbeef.tmp"
        metadata_temp.write_bytes(b"partial metadata")
        upload = self.store.begin("session-a", "audio/ogg")
        Path(upload["upload_path"]).write_bytes(b"partial")
        SharedMediaStore(self.root)
        self.assertFalse(orphan.exists())
        self.assertFalse(metadata_temp.exists())
        self.assertFalse(Path(upload["upload_path"]).exists())

    def test_object_count_quota_is_enforced(self):
        store = SharedMediaStore(self.root / "quota", max_objects=1)
        upload = store.begin("session-a", "audio/ogg")
        Path(upload["upload_path"]).write_bytes(b"OggSone")
        store.commit(upload["upload_id"], "session-a")
        upload = store.begin("session-a", "audio/ogg")
        Path(upload["upload_path"]).write_bytes(b"OggStwo")
        with self.assertRaisesRegex(MediaError, "数量"):
            store.commit(upload["upload_id"], "session-a")


if __name__ == "__main__":
    unittest.main()
