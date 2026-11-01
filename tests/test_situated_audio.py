"""Public synthetic WAV fixtures only; no recording, TTS, playback or IPC."""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import wave

from gateway.display.situated_audio import FixedResponseAudio, KIOSK_OWNER, PUBLIC_TEXTS


class FakeMedia:
    def __init__(self):
        self.imports = []
        self.releases = []
        self.fail_import = False
        self.fail_release = False
        self.release_reply = None
        self.transform = None
        self.during_copy = None

    def create_from_path(self, owner, mime_type, source, **options):
        if self.fail_import:
            raise OSError("synthetic import failure")
        if self.during_copy:
            self.during_copy()
        data = Path(source).read_bytes()
        self.imports.append((owner, mime_type, data, options))
        value = {"media_id": "generated_" + ("%024d" % len(self.imports)),
                 "kind": "generated", "mime_type": "audio/wav", "size": len(data),
                 "created_at": 100, "expires_at": 220}
        return self.transform(value) if self.transform else value

    def call(self, operation, **kwargs):
        if operation != "release":
            raise AssertionError("never resolve or list other media")
        self.releases.append(kwargs)
        if self.fail_release:
            raise OSError("synthetic lost release acknowledgement")
        return self.release_reply if self.release_reply is not None else {
            "media_id": kwargs["media_id"], "released": True}


class FakeOutput:
    def play_media(self, *_args, **_kwargs):
        raise AssertionError("preparation must never play")


class FixedResponseAudioTests(unittest.TestCase):
    def setUp(self):
        # Resolve the test-only temp directory because macOS /var is itself a
        # symlink; the importer deliberately rejects symlinks in every component.
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.media = FakeMedia()
        self.output = FakeOutput()
        self.text = sorted(PUBLIC_TEXTS)[0]
        self.key = hashlib.sha256(self.text.encode()).hexdigest()
        stream = io.BytesIO()
        with wave.open(stream, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(24000)
            writer.writeframes(b"\0\0" * 100)
        self.wav = stream.getvalue()
        self.filename = "feedback-" + self.key[:16] + ".wav"
        self.asset = self.root / self.filename
        self.asset.write_bytes(self.wav)
        self.manifest = {"schema_version": 1, "speed": 1.0, "backend": "melo-qnn-htp",
                         "entries": {self.key: {"file": self.filename,
                                                "sha256": hashlib.sha256(self.wav).hexdigest(),
                                                "size_bytes": len(self.wav)}}}
        self.write_manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def importer(self, **kwargs):
        return FixedResponseAudio(self.media, self.output, asset_root=kwargs.get("asset_root", self.root))

    def test_whitelist_is_exactly_three_public_hints_and_four_fixed_feedbacks(self):
        self.assertEqual(len(PUBLIC_TEXTS), 7)
        self.assertIn("答对啦！要换一题，还是说说你是怎么想到的？", PUBLIC_TEXTS)
        self.assertEqual(KIOSK_OWNER, "session:display-kiosk")

    def test_import_is_generated_owned_bounded_and_never_played(self):
        importer = self.importer()
        self.assertEqual(importer.status()['ready_public_assets'], 1)
        self.assertEqual(self.media.imports, [])  # Verification is not warm-up.
        handle = importer.prepare(self.text)
        self.assertTrue(handle.startswith("generated_"))
        self.assertEqual(self.media.imports, [(KIOSK_OWNER, "audio/wav", self.wav,
                                            {"kind": "generated", "ttl_seconds": 120})])
        self.assertTrue(importer.release(handle))
        self.assertEqual(self.media.releases, [{"owner": KIOSK_OWNER, "media_id": handle}])

    def test_arbitrary_user_text_is_not_imported_or_retained(self):
        importer = self.importer()
        self.assertIsNone(importer.prepare("私人名字和任意对话"))
        self.assertIsNone(importer.prepare(self.text + " "))
        self.assertEqual(self.media.imports, [])
        self.assertNotIn("私人", repr(importer.__dict__))

    def test_non_normal_speed_never_imports_or_generates(self):
        importer = self.importer()
        for speed in (.8, 1.2, True, "1", None, float("nan")):
            with self.subTest(speed=speed):
                self.assertIsNone(importer.prepare(self.text, speed=speed))
        self.assertEqual(self.media.imports, [])

    def test_missing_dependencies_are_safe_fallback(self):
        self.assertIsNone(FixedResponseAudio(None, self.output, self.root).prepare(self.text))
        self.assertIsNone(FixedResponseAudio(self.media, None, self.root).prepare(self.text))
        self.assertEqual(self.media.imports, [])

    def test_handoff_is_exactly_once_and_blocks_late_release_after_uncertain_output(self):
        importer = self.importer()
        handle = importer.prepare(self.text)
        self.assertTrue(importer.handoff(handle))
        self.assertFalse(importer.handoff(handle))
        self.assertFalse(importer.release(handle))
        self.assertEqual(self.media.releases, [])

    def test_foreign_handles_never_release_or_handoff(self):
        importer = self.importer()
        for handle in ("generated_" + "x" * 24, "audio_" + "a" * 24, {}, None):
            with self.subTest(handle=handle):
                self.assertFalse(importer.release(handle))
                self.assertFalse(importer.handoff(handle))
        self.assertEqual(self.media.releases, [])

    def test_at_most_eight_untransferred_imports(self):
        importer = self.importer()
        handles = [importer.prepare(self.text) for _ in range(8)]
        self.assertTrue(all(handles))
        self.assertIsNone(importer.prepare(self.text))
        self.assertEqual(len(self.media.imports), 8)
        self.assertTrue(importer.handoff(handles[0]))
        self.assertIsNotNone(importer.prepare(self.text))

    def test_concurrent_prepares_cannot_exceed_eight_owned_slots(self):
        importer = self.importer()
        handles = []
        threads = [threading.Thread(target=lambda: handles.append(importer.prepare(self.text)))
                   for _ in range(12)]
        for worker in threads:
            worker.start()
        for worker in threads:
            worker.join(2)
        self.assertTrue(all(not worker.is_alive() for worker in threads))
        self.assertEqual(sum(value is not None for value in handles), 8)
        self.assertEqual(len(self.media.imports), 8)

    def test_failed_release_retains_slot_instead_of_forgetting_live_media(self):
        importer = self.importer()
        handles = [importer.prepare(self.text) for _ in range(8)]
        self.media.fail_release = True
        self.assertFalse(importer.release(handles[0]))
        self.assertIsNone(importer.prepare(self.text))
        self.media.fail_release = False
        self.assertTrue(importer.release(handles[0]))
        self.assertIsNotNone(importer.prepare(self.text))

    def test_false_malformed_or_mismatched_release_receipt_retains_owned_slot(self):
        for acknowledgement in ({}, {"released": False}, [],
                                {"released": True, "media_id": "generated_" + "other" * 5}):
            with self.subTest(acknowledgement=acknowledgement):
                self.media = FakeMedia()
                importer = self.importer()
                handles = [importer.prepare(self.text) for _ in range(8)]
                self.media.release_reply = acknowledgement
                self.assertFalse(importer.release(handles[0]))
                self.assertIsNone(importer.prepare(self.text))
                self.media.release_reply = None
                self.assertTrue(importer.release(handles[0]))
                self.assertIsNotNone(importer.prepare(self.text))

    def test_missing_or_corrupt_manifest_and_wav_fall_back(self):
        for bad in (b"not json", b"x" * (16 * 1024 + 1)):
            with self.subTest(bad=len(bad)):
                (self.root / "manifest.json").write_bytes(bad)
                importer = self.importer()
                self.assertEqual(importer.status()['ready_public_assets'], 0)
                self.assertIsNone(importer.prepare(self.text))
        (self.root / "manifest.json").unlink()
        self.assertIsNone(self.importer().prepare(self.text))
        self.write_manifest()
        self.asset.unlink()
        self.assertIsNone(self.importer().prepare(self.text))
        self.asset.write_bytes(b"RIFF" + b"\0" * 100)
        self.assertIsNone(self.importer().prepare(self.text))
        self.assertEqual(self.media.imports, [])

    def test_hash_size_backend_schema_and_speed_must_match(self):
        for mutate in (
            lambda value: value.update(schema_version=True),
            lambda value: value.update(schema_version=2),
            lambda value: value.update(speed=True),
            lambda value: value.update(speed=.9),
            lambda value: value.update(backend="other-voice"),
            lambda value: value["entries"][self.key].update(sha256="0" * 64),
            lambda value: value["entries"][self.key].update(size_bytes=len(self.wav) + 1),
            lambda value: value["entries"][self.key].update(size_bytes=2 * 1024 * 1024 + 1),
        ):
            with self.subTest(mutate=mutate):
                original = copy.deepcopy(self.manifest)
                mutate(self.manifest)
                self.write_manifest()
                self.assertIsNone(self.importer().prepare(self.text))
                self.manifest = original
        self.assertEqual(self.media.imports, [])

    def test_filename_escape_and_unknown_text_key_rejected(self):
        for filename in ("../private.wav", "/tmp/private.wav", "nested/" + self.filename,
                         "feedback-XYZ.wav", "feedback-" + "a" * 16 + ".mp3"):
            with self.subTest(filename=filename):
                self.manifest["entries"][self.key]["file"] = filename
                self.write_manifest()
                self.assertIsNone(self.importer().prepare(self.text))
        self.manifest["entries"] = {"f" * 64: {"file": self.filename, "sha256": "f" * 64, "size_bytes": 100}}
        self.write_manifest()
        self.assertIsNone(self.importer().prepare(self.text))
        self.assertEqual(self.media.imports, [])

    def test_symlink_manifest_asset_and_root_are_rejected_without_reading_targets(self):
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        self.assertIsNone(self.importer(asset_root=alias).prepare(self.text))
        real_manifest = self.root / "saved-manifest.json"
        (self.root / "manifest.json").rename(real_manifest)
        (self.root / "manifest.json").symlink_to(real_manifest)
        self.assertIsNone(self.importer().prepare(self.text))
        (self.root / "manifest.json").unlink()
        real_manifest.rename(self.root / "manifest.json")
        real_asset = self.root / "saved.wav"
        self.asset.rename(real_asset)
        self.asset.symlink_to(real_asset)
        self.assertIsNone(self.importer().prepare(self.text))
        self.assertEqual(self.media.imports, [])

    @unittest.skipUnless(sys.platform.startswith("linux") and hasattr(os, "O_PATH"),
                         "Linux O_PATH search-only directory semantics")
    def test_execute_only_ancestor_with_readable_public_leaves(self):
        ancestor = self.root / "execute-only-ancestor"
        leaf = ancestor / "public-assets"
        leaf.mkdir(parents=True)
        self.asset.rename(leaf / self.filename)
        (self.root / "manifest.json").rename(leaf / "manifest.json")
        ancestor.chmod(0o111)
        try:
            if os.geteuid() != 0:
                # Demonstrate the exact failure the old implementation caused:
                # requesting read permission on this ancestor is unnecessary.
                with self.assertRaises(PermissionError):
                    descriptor = os.open(ancestor, os.O_RDONLY | os.O_DIRECTORY)
                    os.close(descriptor)
            importer = self.importer(asset_root=leaf)
            self.assertEqual(importer.status()["ready_public_assets"], 1)
            handle = importer.prepare(self.text)
            self.assertIsNotNone(handle)
            self.assertEqual(self.media.imports[0][2], self.wav)
            self.assertTrue(importer.release(handle))
            self.assertEqual(stat.S_IMODE(ancestor.stat().st_mode), 0o111)
        finally:
            # Restore only this synthetic test directory for tempfile cleanup.
            ancestor.chmod(0o700)

    def test_hardlinked_or_non_regular_asset_is_rejected(self):
        duplicate = self.root / "hardlink.wav"
        os.link(self.asset, duplicate)
        self.assertIsNone(self.importer().prepare(self.text))
        duplicate.unlink()
        self.asset.unlink()
        self.asset.mkdir()
        self.assertIsNone(self.importer().prepare(self.text))
        self.assertEqual(self.media.imports, [])

    def test_duplicate_manifest_keys_rejected(self):
        raw = json.dumps(self.manifest).replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1')
        (self.root / "manifest.json").write_text(raw, encoding="utf-8")
        self.assertIsNone(self.importer().prepare(self.text))
        self.assertEqual(self.media.imports, [])

    def test_status_has_no_filesystem_or_media_calls(self):
        importer = self.importer()
        with patch("os.open", side_effect=AssertionError("status must not open paths")), \
             patch.object(self.media, "call", side_effect=AssertionError("status must not inspect Media")):
            self.assertEqual(importer.status()['ready_public_assets'], 1)

    def test_invalid_media_reply_is_not_exposed(self):
        for transform in (
            lambda value: {**value, "media_id": "audio_" + "a" * 24},
            lambda value: {**value, "media_id": "generated_short"},
            lambda value: {**value, "kind": "audio"},
            lambda value: {**value, "mime_type": "audio/ogg"},
            lambda value: {**value, "size": value["size"] + 1},
        ):
            with self.subTest(transform=transform):
                self.media = FakeMedia()
                self.media.transform = transform
                self.assertIsNone(self.importer().prepare(self.text))

    def test_path_swap_after_validation_copies_only_original_pinned_public_inode(self):
        importer = self.importer()
        original = self.root / "original.wav"
        replacement = self.root / "not-an-approved-asset.wav"
        replacement.write_bytes(b"MUST_NOT_BE_READ")
        def swap():
            self.asset.rename(original)
            self.asset.symlink_to(replacement)
        self.media.during_copy = swap
        handle = importer.prepare(self.text)
        # Rename changes ctime on some platforms, so a rejected import is also
        # acceptable; either way only the original validated bytes were read.
        self.assertEqual(self.media.imports[0][2], self.wav)
        if handle is not None:
            self.assertTrue(importer.release(handle))

    def test_changed_public_inode_during_copy_is_released_and_not_returned(self):
        importer = self.importer()
        self.media.during_copy = lambda: self.asset.write_bytes(self.wav + b"changed")
        self.assertIsNone(importer.prepare(self.text))
        self.assertEqual(len(self.media.releases), 1)

    def test_import_failure_does_not_start_any_other_action(self):
        importer = self.importer()
        self.media.fail_import = True
        self.assertIsNone(importer.prepare(self.text))
        self.assertEqual(self.media.releases, [])


if __name__ == "__main__":
    unittest.main()
