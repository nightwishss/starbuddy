import tempfile
import unittest
from pathlib import Path

from gateway.media import MediaError, MediaStore


class MediaStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = MediaStore(Path(self.temp.name), ttl_seconds=10, global_limit=1024)

    def tearDown(self):
        self.temp.cleanup()

    def test_create_and_read_owner_media(self):
        item = self.store.create("phone-a", "audio/webm", b"\x1a\x45\xdf\xa3payload")
        self.assertTrue(item.media_id.startswith("audio_"))
        self.assertEqual(self.store.consume(item.media_id, "phone-a"), b"\x1a\x45\xdf\xa3payload")
        self.assertEqual(item.path.stat().st_mode & 0o777, 0o600)
        self.assertNotIn("owner", item.public())

    def test_cross_session_access_is_denied(self):
        item = self.store.create("phone-a", "audio/ogg", b"OggSpayload")
        with self.assertRaisesRegex(MediaError, "无权"):
            self.store.get(item.media_id, "phone-b")

    def test_invalid_signature_and_type_are_rejected(self):
        with self.assertRaises(MediaError):
            self.store.create("phone-a", "audio/webm", b"not-webm")
        with self.assertRaises(MediaError):
            self.store.create("phone-a", "application/octet-stream", b"data")

    def test_expired_media_is_removed(self):
        item = self.store.create("phone-a", "audio/ogg", b"OggSpayload")
        self.assertEqual(self.store.cleanup(item.expires_at + 1), 1)
        self.assertFalse(item.path.exists())

    def test_global_limit_is_enforced(self):
        store = MediaStore(Path(self.temp.name) / "small", global_limit=8)
        with self.assertRaisesRegex(MediaError, "空间不足"):
            store.create("phone-a", "audio/ogg", b"OggS12345")

    def test_jpeg_requires_start_and_end_markers(self):
        with self.assertRaises(MediaError):
            self.store.create("phone-a", "image/jpeg", b"\xff\xd8missing-end")
        item = self.store.create("phone-a", "image/jpeg", b"\xff\xd8pixels\xff\xd9")
        self.assertEqual(item.kind, "image")

    def test_startup_removes_orphans_from_dedicated_root(self):
        root = Path(self.temp.name) / "restart"
        root.mkdir()
        orphan = root / "generated_orphan.wav"
        orphan.write_bytes(b"old session")
        MediaStore(root)
        self.assertFalse(orphan.exists())


if __name__ == "__main__":
    unittest.main()
