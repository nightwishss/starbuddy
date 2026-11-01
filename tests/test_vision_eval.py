#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from tools import vision_eval


class VisionEvaluationTests(unittest.TestCase):
    def test_presence_metrics_report_false_positive_and_missed_label(self):
        samples = [
            {"sample_id": "one", "expected": {"person", "cup"},
             "detections": [{"label": "person", "confidence": 0.9}]},
            {"sample_id": "two", "expected": {"person"},
             "detections": [{"label": "person", "confidence": 0.8},
                            {"label": "chair", "confidence": 0.7}]},
        ]
        result = vision_eval.evaluate(samples, 0.35)
        self.assertEqual(result["micro"]["tp"], 2)
        self.assertEqual(result["micro"]["fp"], 1)
        self.assertEqual(result["micro"]["fn"], 1)
        self.assertEqual(result["failed_sample_count"], 2)

    def test_manifest_is_bounded_and_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            path.write_text(json.dumps({
                "sample_id": "sample-1", "expected_labels": ["person"],
                "detections": [{"label": "person", "confidence": 0.91}],
            }) + "\n", encoding="utf-8")
            samples = vision_eval.load_manifest(path)
        self.assertEqual(samples[0]["expected"], {"person"})

    def test_invalid_confidence_fails_closed(self):
        with self.assertRaises(vision_eval.EvaluationError):
            vision_eval.evaluate([], True)


if __name__ == "__main__":
    unittest.main()
