#!/usr/bin/env python3
"""Metadata-only repeated QNN perception gate with media-leak checks."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CAMERA_OWNER = "service:camera-runtime"


class SoakError(RuntimeError):
    pass


def _media_count(media) -> int | None:
    try:
        value = media.call("list", owner=CAMERA_OWNER)
        return int(value["count"])
    except Exception:
        return None


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return {
        "count": len(values), "p50_ms": round(statistics.median(values), 2),
        "p95_ms": round(p95, 2), "max_ms": round(max(values), 2),
    }


def run(*, samples: int, interval_seconds: float, role="camera.environment",
        profile="fast", confidence=0.35, stability_samples=2,
        perception=None, runtime=None, media=None, clock=time.monotonic,
        sleep=time.sleep) -> dict:
    if isinstance(samples, bool) or not isinstance(samples, int) or not 2 <= samples <= 9600:
        raise SoakError("samples must be between 2 and 9600")
    if not 0 <= interval_seconds <= 60:
        raise SoakError("interval must be between 0 and 60 seconds")
    if perception is None:
        from sdk.perception import PerceptionClient
        perception = PerceptionClient()
    if runtime is None:
        from runtime.client import RuntimeClient
        runtime = RuntimeClient()
    if media is None:
        from media_service import MediaClient
        media = MediaClient()
    media_before = _media_count(media)
    latencies = []
    labels = Counter()
    changes = provisional = failures = consecutive_failures = status_checks = 0
    started = clock()
    for index in range(samples):
        sample_started = clock()
        try:
            result = perception.observe(
                role=role, profile=profile, confidence=confidence,
                stability_samples=stability_samples,
            )
            elapsed = (clock() - sample_started) * 1000
            latencies.append(elapsed)
            stable = result.get("stable_summary") or {}
            labels.update(stable.get("counts") or {})
            change = result.get("change") or {}
            changes += int(change.get("changed") is True)
            provisional += int(change.get("provisional") is True)
            consecutive_failures = 0
        except Exception:
            failures += 1
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break
        if index % 10 == 0:
            response = runtime.call("system.status", {}, timeout_ms=3000)
            status_checks += int(response.get("ok") is True)
        remaining = interval_seconds - (clock() - sample_started)
        if index + 1 < samples and remaining > 0:
            sleep(remaining)
    attempted = len(latencies) + failures
    media_after = _media_count(media)
    distribution = _distribution(latencies)
    media_clean = (
        media_before is not None and media_after is not None and media_after <= media_before
    )
    success_rate = len(latencies) / max(1, attempted)
    passed = (
        attempted == samples and failures == 0 and media_clean
        and distribution["p95_ms"] is not None and distribution["p95_ms"] <= 5000
    )
    return {
        "samples_requested": samples, "samples_attempted": attempted,
        "samples_succeeded": len(latencies), "failures": failures,
        "success_rate": round(success_rate, 4),
        "elapsed_seconds": round(clock() - started, 2),
        "latency": distribution, "changes": changes, "provisional": provisional,
        "stable_label_occurrences": dict(labels.most_common(30)),
        "system_status_checks": status_checks,
        "camera_media_before": media_before, "camera_media_after": media_after,
        "media_clean": media_clean, "passed": passed,
        "raw_pixels_retained": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--role", choices=["camera.front", "camera.environment"],
                        default="camera.environment")
    parser.add_argument("--profile", choices=["fast", "balanced", "detail"], default="fast")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--stability-samples", type=int, choices=[1, 2, 3], default=2)
    arguments = parser.parse_args()
    if not arguments.execute:
        print(json.dumps({
            "ok": True, "status": "dry_run", "samples": arguments.samples,
            "interval_seconds": arguments.interval,
            "estimated_minutes": round(arguments.samples * arguments.interval / 60, 1),
            "raw_pixels_retained": False,
        }, ensure_ascii=False, indent=2))
        return 0
    try:
        result = run(
            samples=arguments.samples, interval_seconds=arguments.interval,
            role=arguments.role, profile=arguments.profile,
            confidence=arguments.confidence, stability_samples=arguments.stability_samples,
        )
    except SoakError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": result["passed"], "result": result}, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
