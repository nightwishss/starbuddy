#!/usr/bin/env python3
"""Read-only stability/resource probe for the pose NPU candidate."""

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


def command(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def unit_value(unit, name):
    return command("systemctl", "--user", "show", unit, "-p", name, "--value")


def process_sample(pid):
    fields = Path(f"/proc/{pid}/stat").read_text().split()
    status = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines()
                  if ":" in line)
    return {
        "at": time.monotonic(),
        "ticks": int(fields[13]) + int(fields[14]),
        "rss_kib": int(status["VmRSS"].split()[0]),
    }


def temperature():
    values = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            values.append(int(path.read_text()) / 1000.0)
        except (OSError, ValueError):
            pass
    return max(values) if values else None


def read_json(url):
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=120)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 10 <= args.duration <= 3600 or args.interval <= 0:
        parser.error("duration must be 10..3600 seconds and interval must be positive")

    units = ("vq-pose-hybrid.service", "vq-pose-npu-worker.service")
    start_restarts = {unit: int(unit_value(unit, "NRestarts")) for unit in units}
    pid = int(unit_value(units[0], "MainPID"))
    if pid <= 0:
        raise SystemExit("candidate host service is not running")
    samples = []
    deadline = time.monotonic() + args.duration
    while True:
        sample = process_sample(pid)
        sample["temperature_c"] = temperature()
        samples.append(sample)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(args.interval, remaining))

    tick_rate = os.sysconf("SC_CLK_TCK")
    cpu = [((right["ticks"] - left["ticks"]) / tick_rate) /
           (right["at"] - left["at"]) * 100.0 for left, right in zip(samples, samples[1:])]
    candidate = read_json("http://127.0.0.1:8792/status")
    worker = read_json("http://127.0.0.1:8793/healthz")
    report = {
        "schema": "vq.pose-npu-soak.v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": args.duration,
        "read_only": True,
        "candidate": candidate,
        "worker": worker,
        "units": {unit: {
            "active": unit_value(unit, "ActiveState"),
            "restart_delta": int(unit_value(unit, "NRestarts")) - start_restarts[unit],
        } for unit in units},
        "host_shell_process": {
            "cpu_percent_one_core_mean": round(statistics.mean(cpu), 2) if cpu else None,
            "cpu_percent_one_core_max": round(max(cpu), 2) if cpu else None,
            "rss_kib_mean": round(statistics.mean(item["rss_kib"] for item in samples)),
            "rss_kib_max": max(item["rss_kib"] for item in samples),
        },
        "temperature_c": {
            "maximum": max(item["temperature_c"] for item in samples
                           if item["temperature_c"] is not None),
        },
        "container": json.loads(command(
            "docker", "inspect", "vq-pose-npu-worker", "--format",
            '{{json .State}}',
        )),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
