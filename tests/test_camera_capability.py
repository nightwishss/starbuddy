#!/usr/bin/env python3
"""USB camera discovery, transform validation and frame lifecycle tests."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from capability.camera import impl
import capability.camera as camera_pkg
from media_service import MediaClientError
from runtime import Registry, execute_capability
from runtime.errors import HardwareError, ResourceNotFoundError, SchemaValidationError


FORMATS = [
    {
        "pixel_format": "mjpeg",
        "description": "Motion-JPEG",
        "sizes": [
            {"width": 640, "height": 480},
            {"width": 1280, "height": 720},
            {"width": 1920, "height": 1080},
        ],
    },
    {
        "pixel_format": "yuyv422",
        "description": "YUYV 4:2:2",
        "sizes": [{"width": 640, "height": 480}],
    },
]


class CameraCapabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.by_id = self.root / "by-id"
        self.frames = self.root / "frames"
        self.devices = self.root / "devices"
        self.by_id.mkdir()
        self.devices.mkdir()
        self.device = self.devices / "video2"
        self.device.write_bytes(b"")
        self.stable_name = "usb-Sonix_RGB_CAMERA_SN0008-video-index0"
        (self.by_id / self.stable_name).symlink_to(self.device)
        self.camera_id = impl._camera_id(self.stable_name)
        self.media_root = self.root / "media"
        self.media_root.mkdir()
        self.media = FakeMediaClient(self.media_root)
        self.root_patches = [
            patch.object(impl, "_BY_ID_ROOT", self.by_id),
            patch.object(impl, "_FRAME_ROOT", self.frames),
            patch.object(impl, "_media_client", return_value=self.media),
        ]
        for item in self.root_patches:
            item.start()
        impl._FRAMES.clear()
        impl._STORE_READY = False

    def tearDown(self):
        impl._FRAMES.clear()
        impl._STORE_READY = False
        for item in reversed(self.root_patches):
            item.stop()
        self.temp.cleanup()

    @patch.object(impl, "_run_format_probe", return_value=FORMATS)
    def test_lists_stable_camera_id(self, _probe):
        result = impl.list_cameras()
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["cameras"][0]["camera_id"], self.camera_id)
        self.assertEqual(result["cameras"][0]["max_width"], 1920)
        self.assertNotEqual(result["cameras"][0]["device"], self.stable_name)

    @patch.object(impl, "_run_format_probe", return_value=FORMATS)
    def test_first_camera_access_removes_orphaned_frames(self, _probe):
        self.frames.mkdir(parents=True)
        orphan = self.frames / "frame_orphan.jpg"
        orphan.write_bytes(b"old")
        impl.list_cameras()
        self.assertFalse(orphan.exists())

    @patch.object(impl, "_query_controls", return_value=[{"name": "Brightness"}])
    @patch.object(impl, "_run_format_probe", return_value=FORMATS)
    def test_get_info_separates_formats_and_controls(self, _probe, _controls):
        result = impl.get_info({"camera_id": self.camera_id})
        self.assertEqual(result["formats"][0]["pixel_format"], "mjpeg")
        self.assertEqual(result["controls"][0]["name"], "Brightness")

    @patch.object(impl, "_run_format_probe", return_value=FORMATS)
    def test_rejects_unsupported_source_size(self, _probe):
        with self.assertRaises(SchemaValidationError):
            impl.snapshot({
                "camera_id": self.camera_id,
                "source": {"pixel_format": "mjpeg", "width": 800, "height": 600},
            })

    @patch.object(impl, "_run_format_probe", return_value=FORMATS)
    def test_rejects_crop_outside_source(self, _probe):
        with self.assertRaises(SchemaValidationError):
            impl.snapshot({
                "camera_id": self.camera_id,
                "source": {"pixel_format": "mjpeg", "width": 640, "height": 480},
                "transform": {"crop": {"x": 500, "y": 0, "width": 200, "height": 200}},
            })

    @patch.object(impl, "_run_format_probe", return_value=FORMATS)
    @patch.object(impl.subprocess, "run")
    def test_snapshot_transform_and_release(self, run, _probe):
        def fake_run(command, **_kwargs):
            output = Path(command[-1])
            output.write_bytes(b"\xff\xd8pixels\xff\xd9")
            return subprocess.CompletedProcess(command, 0, "", "")

        run.side_effect = fake_run
        result = impl.snapshot({
            "camera_id": self.camera_id,
            "source": {"pixel_format": "mjpeg", "width": 1280, "height": 720, "fps": 15},
            "transform": {
                "crop": {"x": 100, "y": 50, "width": 600, "height": 400},
                "rotation": 90,
                "resize": {"width": 320, "height": 240},
            },
            "quality": 80,
        })
        self.assertEqual((result["width"], result["height"]), (320, 240))
        self.assertTrue(result["media_id"].startswith("image_"))
        self.assertTrue(impl.resolve_frame(result["frame_id"]).is_file())
        command = run.call_args.args[0]
        self.assertIn("crop=600:400:100:50,transpose=clock,scale=320:240:flags=lanczos", command)
        self.assertTrue(impl.release_frame({"frame_id": result["frame_id"]})["released"])
        with self.assertRaises(ResourceNotFoundError):
            impl.frame_info({"frame_id": result["frame_id"]})

    @patch.object(impl.HardwareProfileManager, "resolve")
    @patch.object(impl, "_run_format_probe", return_value=FORMATS)
    @patch.object(impl.subprocess, "run")
    def test_snapshot_accepts_logical_camera_role(self, run, _probe, resolve):
        resolve.return_value = {"device": {"stable_id": self.stable_name}}
        run.side_effect = lambda command, **kwargs: _fake_ffmpeg(command)
        registry = Registry()
        camera_pkg.register(registry)
        result = execute_capability(registry, "camera.snapshot", {"role": "camera.front", "source": {
            "pixel_format": "mjpeg", "width": 640, "height": 480}})
        self.assertEqual(result["camera_id"], self.camera_id)

    def test_snapshot_requires_exactly_one_identity(self):
        with self.assertRaises(SchemaValidationError):
            impl.snapshot({})
        with self.assertRaises(SchemaValidationError):
            impl.snapshot({"camera_id": self.camera_id, "role": "camera.front"})

    @patch.object(impl, "_run_format_probe", return_value=FORMATS)
    @patch.object(impl.subprocess, "run")
    def test_media_service_failure_is_explicit_and_cleans_staging(self, run, _probe):
        run.side_effect = lambda command, **kwargs: _fake_ffmpeg(command)
        with patch.object(
                self.media, "create_from_path",
                side_effect=MediaClientError("UNAVAILABLE", "down")):
            with self.assertRaises(HardwareError):
                impl.snapshot({"camera_id": self.camera_id, "source": {
                    "pixel_format": "mjpeg", "width": 640, "height": 480}})
        self.assertEqual(list(self.frames.glob("*.jpg")), [])


def _fake_ffmpeg(command):
    output = Path(command[-1])
    output.write_bytes(b"\xff\xd8pixels\xff\xd9")
    return subprocess.CompletedProcess(command, 0, "", "")


class FakeMediaClient:
    def __init__(self, root):
        self.root = root
        self.items = {}

    def create_from_path(self, owner, mime_type, source, ttl_seconds=None):
        token = f"image_{'x' * 24}{len(self.items)}"
        path = self.root / f"{token}.jpg"
        path.write_bytes(Path(source).read_bytes())
        item = {"media_id": token, "size": path.stat().st_size,
                "expires_at": 9999999999.0, "path": path, "owner": owner}
        self.items[token] = item
        return item

    def call(self, op, owner, media_id):
        item = self.items.get(media_id)
        if item is None or item["owner"] != owner:
            raise RuntimeError("not found")
        if op == "resolve":
            return {"media_id": media_id, "path": str(item["path"])}
        if op == "release":
            item["path"].unlink(missing_ok=True)
            self.items.pop(media_id, None)
            return {"released": True}
        raise RuntimeError("unsupported")


if __name__ == "__main__":
    unittest.main()
