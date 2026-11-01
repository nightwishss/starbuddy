"""Read-only Linux/cgroup telemetry with explicit hardware-health semantics."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


SEVERITY = {"ok": 0, "warning": 1, "critical": 2}
CRITICAL_UNITS = (
    "vq-display.service",
    "vq-display-kiosk.service",
    "vq-camera-stream.service",
    "vq-runtime.service",
    "vq-local-llm.service",
    "vq-output.service",
    "vq-task-orchestrator.service",
    "vq-task-interaction.service",
    "vq-pose-npu.service",
    "vq-face.service",
    "vq-face-identity.service",
    "vq-perception-context.service",
    "vq-affect-cues.service",
    "vq-events.service",
)


def _text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _integer(path: Path) -> int | None:
    value = _text(path)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


class TelemetryCollector:
    def __init__(self, *, proc_root=Path("/proc"), sys_root=Path("/sys"),
                 cgroup_root=None, disk_root=Path("/"), clock=time.monotonic,
                 wall_clock=time.time, command=subprocess.run,
                 disk_usage=shutil.disk_usage, critical_units=CRITICAL_UNITS):
        self.proc_root = Path(proc_root)
        self.sys_root = Path(sys_root)
        self.cgroup_root = Path(cgroup_root or self.sys_root / (
            "fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice"
        ))
        self.disk_root = Path(disk_root)
        self.clock = clock
        self.wall_clock = wall_clock
        self.command = command
        self.disk_usage = disk_usage
        self.critical_units = tuple(critical_units)
        self.previous_cpu: tuple[int, int] | None = None
        self.previous_apps: dict[str, tuple[int, int, int]] = {}
        self.previous_at: float | None = None
        self.container_cgroups: dict[str, Path] = {}
        self.unit_states_cache: dict[str, dict] = {}
        self.unit_names_cache: tuple[str, ...] = ()
        self.unit_states_at = float("-inf")
        self.max_temperature = float("-inf")
        self.over_temperature_seconds = 0.0

    def _cpu(self) -> dict:
        line = (_text(self.proc_root / "stat") or "").splitlines()
        values = [int(value) for value in line[0].split()[1:]] if line else []
        total = sum(values)
        idle = sum(values[index] for index in (3, 4) if index < len(values))
        percent = None
        if self.previous_cpu is not None:
            delta_total = total - self.previous_cpu[0]
            delta_idle = idle - self.previous_cpu[1]
            if delta_total > 0:
                percent = round(100 * (delta_total - delta_idle) / delta_total, 2)
        self.previous_cpu = (total, idle)
        try:
            load = [round(value, 3) for value in os.getloadavg()]
        except OSError:
            load = []
        uptime_text = _text(self.proc_root / "uptime") or "0"
        try:
            uptime = float(uptime_text.split()[0])
        except (ValueError, IndexError):
            uptime = 0.0
        return {"percent": percent, "cores": os.cpu_count() or 1,
                "load_1_5_15": load, "uptime_seconds": round(uptime, 1)}

    def _memory(self) -> dict:
        values = {}
        for line in (_text(self.proc_root / "meminfo") or "").splitlines():
            name, _, remainder = line.partition(":")
            try:
                values[name] = int(remainder.strip().split()[0]) * 1024
            except (ValueError, IndexError):
                continue
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", 0)
        used = max(0, total - available)
        return {"total_bytes": total, "used_bytes": used, "available_bytes": available,
                "used_percent": round(100 * used / total, 2) if total else None}

    def _disk(self) -> dict:
        try:
            usage = self.disk_usage(self.disk_root)
        except OSError:
            return {"available": False}
        return {"available": True, "total_bytes": usage.total, "used_bytes": usage.used,
                "free_bytes": usage.free, "used_percent": round(100 * usage.used / usage.total, 2)}

    def _thermal(self, elapsed: float) -> dict:
        sensors = []
        for zone in sorted((self.sys_root / "class/thermal").glob("thermal_zone*")):
            raw = _integer(zone / "temp")
            if raw is None:
                continue
            celsius = raw / 1000 if abs(raw) > 500 else float(raw)
            sensors.append({"name": _text(zone / "type") or zone.name,
                            "celsius": round(celsius, 2)})
        sensors.sort(key=lambda item: item["celsius"], reverse=True)
        maximum = sensors[0]["celsius"] if sensors else None
        if maximum is not None:
            self.max_temperature = max(self.max_temperature, maximum)
            if maximum >= 75:
                self.over_temperature_seconds += max(0.0, elapsed)
        return {"max_celsius": maximum,
                "max_observed_celsius": None if self.max_temperature == float("-inf")
                else round(self.max_temperature, 2),
                "seconds_at_or_above_75c": round(self.over_temperature_seconds, 1),
                "hottest_sensors": sensors[:6]}

    def _storage_health(self) -> dict:
        block = self.sys_root / "class/block/mmcblk0"
        life_raw = _text(block / "device/life_time")
        pre_eol_raw = _text(block / "device/pre_eol_info")
        life_values = []
        for value in (life_raw or "").split():
            try:
                life_values.append(int(value, 16))
            except ValueError:
                pass
        try:
            pre_eol = int(pre_eol_raw, 16) if pre_eol_raw else None
        except ValueError:
            pre_eol = None
        stat = (_text(block / "stat") or "").split()
        try:
            sectors_written = int(stat[6])
        except (ValueError, IndexError):
            sectors_written = None
        maximum = max(life_values, default=None)
        if maximum is None:
            estimate = None
        elif maximum == 0:
            estimate = "not_reported"
        elif maximum <= 10:
            estimate = f"{(maximum - 1) * 10}-{maximum * 10}%_used"
        else:
            estimate = "exceeded_estimated_lifetime"
        return {
            "device": "mmcblk0", "life_time_raw": life_raw,
            "estimated_life_used_range": estimate,
            "pre_eol_raw": pre_eol_raw,
            "pre_eol_state": {1: "normal", 2: "warning", 3: "urgent"}.get(pre_eol, "unknown"),
            "bytes_written_since_boot": None if sectors_written is None else sectors_written * 512,
            "measurement_note": "device self-report and boot counters; not a failure-date prediction",
        }

    def _typec_health(self) -> dict:
        port = self.sys_root / "class/typec/port0"
        touch_names = []
        for event in sorted((self.sys_root / "class/input").glob("event*")):
            name = _text(event / "device/name")
            if name and "touch" in name.lower() and name not in touch_names:
                touch_names.append(name[:80])
        return {
            "port0_present": port.exists(),
            "data_role": _text(port / "data_role"),
            "power_role": _text(port / "power_role"),
            "touch_input_present": bool(touch_names),
            "touch_devices": touch_names[:4],
            "recovery_policy": "kernel_reenumeration; boot-only bounded UCSI rebind",
        }

    @staticmethod
    def _io(path: Path) -> tuple[int, int]:
        read_bytes = write_bytes = 0
        for line in (_text(path) or "").splitlines():
            for token in line.split()[1:]:
                key, _, value = token.partition("=")
                try:
                    amount = int(value)
                except ValueError:
                    continue
                if key == "rbytes":
                    read_bytes += amount
                elif key == "wbytes":
                    write_bytes += amount
        return read_bytes, write_bytes

    def _container_paths(self) -> dict[str, Path]:
        if self.container_cgroups and all(path.is_dir() for path in self.container_cgroups.values()):
            return self.container_cgroups
        names = ("vq-local-llm", "vq-vision-runner")
        try:
            completed = self.command(
                ["docker", "inspect", "--format", "{{.Name}} {{.Id}}", *names],
                capture_output=True, text=True, timeout=2, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}
        values = {}
        for line in completed.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            name, container_id = parts[0].lstrip("/"), parts[1]
            if name not in names or len(container_id) < 12:
                continue
            path = self.sys_root / "fs/cgroup/system.slice" / f"docker-{container_id}.scope"
            if path.is_dir():
                values[name] = path
        self.container_cgroups = values
        return values

    def _raw_cgroup(self, path: Path) -> tuple[int, int, int, int | None, int | None]:
        cpu_values = {}
        for line in (_text(path / "cpu.stat") or "").splitlines():
            key, _, value = line.partition(" ")
            try:
                cpu_values[key] = int(value)
            except ValueError:
                pass
        read_bytes, write_bytes = self._io(path / "io.stat")
        return (cpu_values.get("usage_usec", 0), read_bytes, write_bytes,
                _integer(path / "memory.current"), _integer(path / "pids.current"))

    def _unit_states(self, units: list[str], now: float) -> dict[str, dict]:
        if not units:
            return {}
        names = tuple(units)
        if names == self.unit_names_cache and now - self.unit_states_at < 30:
            return self.unit_states_cache
        try:
            completed = self.command(
                ["systemctl", "--user", "show", *units, "--property=Id",
                 "--property=LoadState", "--property=ActiveState", "--property=SubState",
                 "--property=NRestarts", "--property=MainPID"],
                capture_output=True, text=True, timeout=3, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return {}
        result = {}
        for block in completed.stdout.strip().split("\n\n"):
            values = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
            unit = values.get("Id")
            if unit:
                result[unit] = {
                    "load_state": values.get("LoadState", "unknown"),
                    "active_state": values.get("ActiveState", "unknown"),
                    "sub_state": values.get("SubState", "unknown"),
                    "restarts": int(values.get("NRestarts", "0") or 0),
                    "main_pid": int(values.get("MainPID", "0") or 0) or None,
                }
        self.unit_states_cache = result
        self.unit_names_cache = names
        self.unit_states_at = now
        return result

    def _applications(self, now: float, elapsed: float) -> dict:
        discovered = {
            path.name: path for path in self.cgroup_root.glob("vq-*.service")
        } if self.cgroup_root.is_dir() else {}
        unit_names = sorted(set(discovered) | set(self.critical_units))
        states = self._unit_states(unit_names, now)
        containers = self._container_paths()
        result = {}
        next_previous = {}
        for unit_name in unit_names:
            path = discovered.get(unit_name)
            name = unit_name.removesuffix(".service")
            if path is None:
                usage = read_bytes = write_bytes = 0
                memory = pids = None
            else:
                usage, read_bytes, write_bytes, memory, pids = self._raw_cgroup(path)
            container = containers.get(name)
            if container is not None:
                child_usage, child_read, child_write, child_memory, child_pids = self._raw_cgroup(container)
                usage += child_usage
                read_bytes += child_read
                write_bytes += child_write
                memory = (memory or 0) + (child_memory or 0)
                pids = (pids or 0) + (child_pids or 0)
            previous = self.previous_apps.get(unit_name)
            cpu_percent = read_rate = write_rate = None
            if path is not None and previous is not None and elapsed > 0:
                cpu_percent = round(max(0, usage - previous[0]) / (elapsed * 1_000_000) * 100, 2)
                read_rate = round(max(0, read_bytes - previous[1]) / elapsed, 2)
                write_rate = round(max(0, write_bytes - previous[2]) / elapsed, 2)
            if path is not None:
                next_previous[unit_name] = (usage, read_bytes, write_bytes)
            state = states.get(unit_name, {
                "load_state": "unknown", "active_state": "unknown",
                "sub_state": "unknown", "restarts": 0, "main_pid": None,
            })
            active = state["active_state"]
            alive = None if active == "unknown" else active == "active"
            ready = None if alive is None else alive and state["sub_state"] in {"running", "exited"}
            result[name] = {
                **state,
                "alive": alive,
                "ready": ready,
                "cpu_percent": cpu_percent,
                "memory_bytes": memory,
                "pids": pids,
                "read_bytes_per_second": read_rate,
                "write_bytes_per_second": write_rate,
                "resource_scope": "service+container" if container is not None else "service",
            }
        self.previous_apps = next_previous
        return result

    @staticmethod
    def _alerts(system: dict, hardware: dict, applications: dict) -> list[dict]:
        alerts = []
        def add(severity, code, message, measured):
            alerts.append({"severity": severity, "code": code, "message": message,
                           "measured": measured})
        temperature = system["thermal"].get("max_celsius")
        if temperature is not None and temperature >= 85:
            add("critical", "temperature", "设备温度过高", temperature)
        elif temperature is not None and temperature >= 75:
            add("warning", "temperature", "设备温度偏高", temperature)
        memory = system["memory"].get("used_percent")
        if memory is not None and memory >= 95:
            add("critical", "memory", "可用内存严重不足", memory)
        elif memory is not None and memory >= 90:
            add("warning", "memory", "可用内存偏低", memory)
        disk = system["disk"].get("used_percent")
        if disk is not None and disk >= 95:
            add("critical", "disk", "系统盘空间严重不足", disk)
        elif disk is not None and disk >= 85:
            add("warning", "disk", "系统盘空间偏低", disk)
        pre_eol = hardware["storage"].get("pre_eol_state")
        if pre_eol in {"warning", "urgent"}:
            add("critical" if pre_eol == "urgent" else "warning", "storage_pre_eol",
                "存储器报告预失效状态", pre_eol)
        for name, app in applications.items():
            if app.get("active_state") not in {"active", "unknown"}:
                add("critical", "service_inactive", f"{name} 服务未运行", app.get("active_state"))
            elif app.get("restarts", 0) >= 3:
                add("warning", "service_restarts", f"{name} 重启次数偏多", app["restarts"])
        return alerts

    def sample(self) -> dict:
        now = self.clock()
        elapsed = 0.0 if self.previous_at is None else max(0.0, now - self.previous_at)
        self.previous_at = now
        system = {"cpu": self._cpu(), "memory": self._memory(), "disk": self._disk()}
        system["thermal"] = self._thermal(elapsed)
        hardware = {"storage": self._storage_health(), "typec": self._typec_health()}
        applications = self._applications(now, elapsed)
        alerts = self._alerts(system, hardware, applications)
        severity = max((item["severity"] for item in alerts),
                       key=lambda item: SEVERITY[item], default="ok")
        return {"timestamp": self.wall_clock(), "status": severity, "alerts": alerts,
                "system": system, "hardware": hardware, "applications": applications}
