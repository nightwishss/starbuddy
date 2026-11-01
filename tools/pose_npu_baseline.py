#!/usr/bin/env python3
"""Capture a read-only pose-runtime baseline for CPU/NPU comparisons.

The tool only reads loopback HTTP, /proc, sysfs, and systemd properties. It
does not stop, restart, enable, or modify any service or model.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def read_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.load(response)


def systemctl_value(unit: str, prop: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", prop, "--value"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def process_sample(pid: int) -> dict:
    stat = Path(f"/proc/{pid}/stat").read_text().split()
    status = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            status[key] = value.strip()
    return {
        "monotonic_s": time.monotonic(),
        "cpu_ticks": int(stat[13]) + int(stat[14]),
        "rss_kib": int(status["VmRSS"].split()[0]),
        "threads": int(status["Threads"]),
    }


def thermal_snapshot() -> list[dict]:
    values = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            millidegrees = int(path.read_text().strip())
        except (OSError, ValueError):
            continue
        type_path = path.parent / "type"
        try:
            kind = type_path.read_text().strip()
        except OSError:
            kind = path.parent.name
        values.append({"zone": path.parent.name, "type": kind, "celsius": millidegrees / 1000})
    return sorted(values, key=lambda item: item["celsius"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status-url", default="http://127.0.0.1:8791/status")
    parser.add_argument("--unit", default="vq-pose-runner.service")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.duration <= 0 or args.interval <= 0:
        parser.error("duration and interval must be positive")

    pid = int(systemctl_value(args.unit, "MainPID"))
    if pid <= 0:
        raise SystemExit(f"{args.unit} has no running MainPID")

    first = process_sample(pid)
    samples = [first]
    deadline = time.monotonic() + args.duration
    while time.monotonic() < deadline:
        time.sleep(min(args.interval, max(0, deadline - time.monotonic())))
        samples.append(process_sample(pid))

    clock_ticks = os.sysconf("SC_CLK_TCK")
    cpu_percent = []
    for left, right in zip(samples, samples[1:]):
        elapsed = right["monotonic_s"] - left["monotonic_s"]
        ticks = right["cpu_ticks"] - left["cpu_ticks"]
        cpu_percent.append((ticks / clock_ticks) / elapsed * 100)

    status = read_json(args.status_url)
    report = {
        "schema": "vq.pose-baseline.v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "unit": {
            "name": args.unit,
            "active_state": systemctl_value(args.unit, "ActiveState"),
            "enabled_state": subprocess.run(
                ["systemctl", "--user", "is-enabled", args.unit],
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "pid": pid,
            "restart_count": int(systemctl_value(args.unit, "NRestarts")),
        },
        "runtime_status": status,
        "process": {
            "duration_s": args.duration,
            "sample_count": len(samples),
            "cpu_percent_one_core": {
                "mean": round(statistics.mean(cpu_percent), 2),
                "median": round(statistics.median(cpu_percent), 2),
                "max": round(max(cpu_percent), 2),
            },
            "rss_kib": {
                "mean": round(statistics.mean(item["rss_kib"] for item in samples)),
                "max": max(item["rss_kib"] for item in samples),
            },
            "threads_max": max(item["threads"] for item in samples),
        },
        "thermal_top": thermal_snapshot()[:8],
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
