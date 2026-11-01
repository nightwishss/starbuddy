#!/usr/bin/env python3

import asyncio
import json
import subprocess
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import MagicMock

from telemetry_service.collector import TelemetryCollector
from telemetry_service.service import TelemetryServer, TelemetryStore


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


class TelemetryCollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.proc = self.root / "proc"
        self.sys = self.root / "sys"
        self.cgroups = self.root / "cgroups"
        self.proc.mkdir()
        self.cgroups.mkdir()
        (self.proc / "stat").write_text("cpu  100 0 50 850 0 0 0 0\n")
        (self.proc / "uptime").write_text("1000.0 0\n")
        (self.proc / "meminfo").write_text("MemTotal: 1000 kB\nMemAvailable: 400 kB\n")
        thermal = self.sys / "class/thermal/thermal_zone0"
        thermal.mkdir(parents=True)
        (thermal / "type").write_text("cpu-test\n")
        (thermal / "temp").write_text("76000\n")
        block = self.sys / "class/block/mmcblk0/device"
        block.mkdir(parents=True)
        (block / "life_time").write_text("0x01 0x01\n")
        (block / "pre_eol_info").write_text("0x01\n")
        (block.parent / "stat").write_text("1 0 2 0 3 0 4 0 0 0 0\n")
        typec = self.sys / "class/typec/port0"
        typec.mkdir(parents=True)
        (typec / "data_role").write_text("[host] device\n")
        (typec / "power_role").write_text("source [sink]\n")
        touch = self.sys / "class/input/event7/device"
        touch.mkdir(parents=True)
        (touch / "name").write_text("Test TouchScreen\n")
        duplicate_touch = self.sys / "class/input/event8/device"
        duplicate_touch.mkdir(parents=True)
        (duplicate_touch / "name").write_text("Test TouchScreen\n")
        app = self.cgroups / "vq-test.service"
        app.mkdir()
        (app / "cpu.stat").write_text("usage_usec 1000000\n")
        (app / "memory.current").write_text("2048\n")
        (app / "pids.current").write_text("2\n")
        (app / "io.stat").write_text("8:0 rbytes=100 wbytes=200\n")
        self.clock = Clock()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def command(_arguments, **_kwargs):
        return subprocess.CompletedProcess([], 0, (
            "Id=vq-test.service\nLoadState=loaded\nActiveState=active\n"
            "SubState=running\nNRestarts=1\nMainPID=123\n"
        ), "")

    def test_samples_system_application_and_reported_storage_health(self):
        usage = namedtuple("usage", "total used free")
        collector = TelemetryCollector(
            proc_root=self.proc, sys_root=self.sys, cgroup_root=self.cgroups,
            disk_root=self.root, clock=self.clock, wall_clock=lambda: 123.0,
            command=self.command, disk_usage=lambda _path: usage(1000, 500, 500),
        )
        first = collector.sample()
        self.clock.value += 5
        (self.proc / "stat").write_text("cpu  140 0 60 900 0 0 0 0\n")
        app = self.cgroups / "vq-test.service"
        (app / "cpu.stat").write_text("usage_usec 1500000\n")
        (app / "io.stat").write_text("8:0 rbytes=600 wbytes=1200\n")
        second = collector.sample()

        self.assertEqual(first["hardware"]["storage"]["pre_eol_state"], "normal")
        self.assertEqual(first["hardware"]["storage"]["estimated_life_used_range"], "0-10%_used")
        self.assertTrue(first["hardware"]["typec"]["port0_present"])
        self.assertTrue(first["hardware"]["typec"]["touch_input_present"])
        self.assertEqual(first["hardware"]["typec"]["data_role"], "[host] device")
        self.assertEqual(first["hardware"]["typec"]["touch_devices"], ["Test TouchScreen"])
        self.assertEqual(second["applications"]["vq-test"]["cpu_percent"], 10.0)
        self.assertEqual(second["applications"]["vq-test"]["write_bytes_per_second"], 200.0)
        self.assertTrue(second["applications"]["vq-test"]["alive"])
        self.assertTrue(second["applications"]["vq-test"]["ready"])
        self.assertEqual(second["applications"]["vq-test"]["main_pid"], 123)
        self.assertEqual(second["system"]["disk"]["used_percent"], 50.0)
        self.assertEqual(second["system"]["thermal"]["seconds_at_or_above_75c"], 5.0)
        self.assertEqual(second["status"], "warning")

    def test_critical_services_are_reported_without_active_cgroups(self):
        usage = namedtuple("usage", "total used free")
        collector = TelemetryCollector(
            proc_root=self.proc, sys_root=self.sys, cgroup_root=self.cgroups,
            disk_root=self.root, clock=self.clock, wall_clock=lambda: 123.0,
            command=self.command, disk_usage=lambda _path: usage(1000, 500, 500),
        )
        applications = collector.sample()["applications"]
        for name in (
            "vq-display", "vq-display-kiosk", "vq-camera-stream", "vq-runtime",
            "vq-local-llm", "vq-output", "vq-task-orchestrator", "vq-pose-npu",
            "vq-task-interaction", "vq-face", "vq-face-identity", "vq-events",
            "vq-perception-context",
        ):
            self.assertIn(name, applications)
        self.assertIsNone(applications["vq-display"]["alive"])
        self.assertIsNone(applications["vq-display"]["ready"])

    def test_inactive_critical_service_remains_visible_and_alerts(self):
        usage = namedtuple("usage", "total used free")
        def inactive(_arguments, **_kwargs):
            return subprocess.CompletedProcess([], 0, (
                "Id=vq-display.service\nLoadState=loaded\nActiveState=inactive\n"
                "SubState=dead\nNRestarts=2\nMainPID=0\n"
            ), "")
        collector = TelemetryCollector(
            proc_root=self.proc, sys_root=self.sys, cgroup_root=self.cgroups,
            disk_root=self.root, clock=self.clock, wall_clock=lambda: 123.0,
            command=inactive, disk_usage=lambda _path: usage(1000, 500, 500),
            critical_units=("vq-display.service",),
        )
        sample = collector.sample()
        display = sample["applications"]["vq-display"]
        self.assertFalse(display["alive"])
        self.assertFalse(display["ready"])
        self.assertIsNone(display["memory_bytes"])
        self.assertIn("service_inactive", {item["code"] for item in sample["alerts"]})

    def test_alerts_distinguish_measured_thresholds(self):
        alerts = TelemetryCollector._alerts(
            {"thermal": {"max_celsius": 86}, "memory": {"used_percent": 96},
             "disk": {"used_percent": 90}},
            {"storage": {"pre_eol_state": "urgent"}},
            {"bad": {"active_state": "failed", "restarts": 4}},
        )
        self.assertIn("critical", {item["severity"] for item in alerts})
        self.assertIn("storage_pre_eol", {item["code"] for item in alerts})
        self.assertIn("service_inactive", {item["code"] for item in alerts})


class TelemetryStoreTests(unittest.TestCase):
    def test_history_is_bounded(self):
        store = TelemetryStore(maximum=2)
        for value in range(3):
            store.append({"value": value})
        self.assertEqual(store.latest()["value"], 2)
        self.assertEqual([item["value"] for item in store.history(2)], [1, 2])
        with self.assertRaises(ValueError):
            store.history(0)


class TelemetryServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_private_socket_serves_latest_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            socket_path = Path(temporary) / "telemetry.sock"
            collector = MagicMock()
            collector.sample.return_value = {"timestamp": 1, "status": "ok"}
            server = TelemetryServer(
                socket_path=socket_path, interval=0.02, collector=collector,
            )
            await server.start()
            await asyncio.sleep(0.04)
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            writer.write(b'{"op":"latest"}\n')
            await writer.drain()
            response = json.loads(await reader.readline())
            writer.close()
            await writer.wait_closed()
            await server.close()
        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
