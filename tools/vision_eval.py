#!/usr/bin/env python3
"""Evaluate structured object-presence results from an operator-labelled JSONL manifest."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


class EvaluationError(ValueError):
    pass


def _labels(value, field: str) -> set[str]:
    if not isinstance(value, list) or len(value) > 100:
        raise EvaluationError(f"{field} must be a bounded array")
    result = set()
    for label in value:
        if not isinstance(label, str) or not label or len(label) > 80:
            raise EvaluationError(f"{field} contains an invalid label")
        result.add(label)
    return result


def load_manifest(path: Path) -> list[dict]:
    samples = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvaluationError("manifest cannot be read") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"invalid JSON on line {line_number}") from exc
        if not isinstance(item, dict) or not isinstance(item.get("sample_id"), str):
            raise EvaluationError(f"invalid sample on line {line_number}")
        expected = _labels(item.get("expected_labels"), "expected_labels")
        detections = item.get("detections")
        if not isinstance(detections, list) or len(detections) > 500:
            raise EvaluationError("detections must be a bounded array")
        normalized = []
        for detection in detections:
            if (
                not isinstance(detection, dict)
                or not isinstance(detection.get("label"), str)
                or isinstance(detection.get("confidence"), bool)
                or not isinstance(detection.get("confidence"), (int, float))
                or not 0 <= detection["confidence"] <= 1
            ):
                raise EvaluationError(f"invalid detection on line {line_number}")
            normalized.append({
                "label": detection["label"], "confidence": float(detection["confidence"]),
            })
        samples.append({"sample_id": item["sample_id"], "expected": expected,
                        "detections": normalized})
        if len(samples) > 10_000:
            raise EvaluationError("manifest exceeds 10000 samples")
    if not samples:
        raise EvaluationError("manifest is empty")
    return samples


def evaluate(samples: list[dict], confidence=0.35) -> dict:
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise EvaluationError("confidence must be between 0 and 1")
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    failures = []
    for sample in samples:
        expected = sample["expected"]
        predicted = {
            item["label"] for item in sample["detections"] if item["confidence"] >= confidence
        }
        for label in expected | predicted:
            if label in expected and label in predicted:
                counts[label]["tp"] += 1
            elif label in predicted:
                counts[label]["fp"] += 1
            else:
                counts[label]["fn"] += 1
        if expected != predicted and len(failures) < 100:
            failures.append({
                "sample_id": sample["sample_id"],
                "missed": sorted(expected - predicted),
                "unexpected": sorted(predicted - expected),
            })

    per_label = {}
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for label, value in sorted(counts.items()):
        for key in totals:
            totals[key] += value[key]
        precision = value["tp"] / max(1, value["tp"] + value["fp"])
        recall = value["tp"] / max(1, value["tp"] + value["fn"])
        per_label[label] = {
            **value, "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / max(1e-12, precision + recall), 4),
        }
    precision = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    recall = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    return {
        "samples": len(samples), "confidence": confidence,
        "micro": {
            **totals, "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / max(1e-12, precision + recall), 4),
        },
        "per_label": per_label, "failed_samples": failures,
        "failed_sample_count": sum(
            sample["expected"] != {
                item["label"] for item in sample["detections"]
                if item["confidence"] >= confidence
            } for sample in samples
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--minimum-precision", type=float, default=0.0)
    parser.add_argument("--minimum-recall", type=float, default=0.0)
    parser.add_argument("--example", action="store_true")
    arguments = parser.parse_args()
    if arguments.example:
        print(json.dumps({
            "sample_id": "operator-defined-id", "expected_labels": ["person", "cup"],
            "detections": [{"label": "person", "confidence": 0.91}],
        }, ensure_ascii=False))
        return 0
    if arguments.manifest is None:
        parser.error("manifest is required unless --example is used")
    try:
        result = evaluate(load_manifest(arguments.manifest), arguments.confidence)
    except EvaluationError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 2
    passed = (
        result["micro"]["precision"] >= arguments.minimum_precision
        and result["micro"]["recall"] >= arguments.minimum_recall
    )
    print(json.dumps({"ok": passed, "result": result}, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
