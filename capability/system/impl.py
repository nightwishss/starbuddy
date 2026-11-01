"""System Capability 实现 (Query 型) — Capability Contract v0.1.

底层真实接口 (纯软件, 只读 /proc + /sys + stdlib, 无外部依赖):
  /proc/loadavg /proc/meminfo /proc/uptime /proc/stat
  /sys/class/thermal/thermal_zone*/temp
  os.uname() / platform / shutil.disk_usage

v0.1 只封装只读查询。服务启停 (systemctl start/stop) 需 sudo,
留待 Phase 3 权限层再决定是否暴露 (arduino 用户 NOPASSWD 仅限少数命令)。
"""
import glob
import os
import platform
import shutil
import time

from runtime.errors import HardwareError


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError as e:
        raise HardwareError(
            f"读 {path} 失败: {e!r}", capability="system", details={"path": path}) from e


def _loadavg():
    parts = _read("/proc/loadavg").split()
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def _cpu_times():
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu "):
                # user nice system idle iowait irq softirq steal ...
                return [int(x) for x in line.split()[1:]]
    raise HardwareError("未找到 /proc/stat cpu 行", capability="system")


def _cpu_percent():
    t1 = _cpu_times()
    time.sleep(0.2)
    t2 = _cpu_times()
    idle1, total1 = t1[3], sum(t1)
    idle2, total2 = t2[3], sum(t2)
    d_idle = idle2 - idle1
    d_total = total2 - total1
    if d_total <= 0:
        return 0.0
    return round(100.0 * (1.0 - d_idle / d_total), 1)


def _meminfo():
    d = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            d[k.strip()] = int(v.strip().split()[0])  # kB
    total = d.get("MemTotal", 0)
    avail = d.get("MemAvailable", d.get("MemFree", 0))
    used = total - avail
    return total, avail, used


def _temp_max():
    temps = []
    for z in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            temps.append(int(_read(z)) / 1000.0)
        except HardwareError:
            continue
    return max(temps) if temps else 0.0  # 0.0 = 无传感器 (板子实际有 23 个 zone)


def status(args=None):
    total, avail, used = _meminfo()
    du = shutil.disk_usage("/")
    up = float(_read("/proc/uptime").split()[0])
    return {
        "loadavg": _loadavg(),
        "cpu_percent": _cpu_percent(),
        "cpu_count": os.cpu_count(),
        "mem": {
            "total_kb": total, "available_kb": avail, "used_kb": used,
            "percent": round(100.0 * used / total, 1) if total else 0.0,
        },
        "disk": {
            "total_gb": round(du.total / 1e9, 2), "used_gb": round(du.used / 1e9, 2),
            "free_gb": round(du.free / 1e9, 2),
            "percent": round(100.0 * du.used / du.total, 1),
        },
        "uptime_seconds": int(up),
        "boot_time": int(time.time() - up),
        "temp_max_c": _temp_max(),
    }


def info(args=None):
    return {
        "hostname": platform.node(),
        "system": platform.system(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "python": platform.python_version(),
    }


# action 名 -> 实现函数 (由 capability/system/__init__.py 注册进 Registry)
ACTIONS = {
    "system.status": status,
    "system.info": info,
}
