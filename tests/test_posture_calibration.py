#!/usr/bin/env python3

import unittest

from perception import assess_posture, build_posture_baseline


def feature(head=.2, height=.8, slope=.04, width=.3, torso=4.0, confidence=.95):
    return {"confidence": confidence, "values": {
        "head_forward_ratio": head, "head_height_ratio": height,
        "shoulder_slope_ratio": slope, "shoulder_width": width,
        "torso_lean_degrees": torso,
    }}


class PostureCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.baseline = build_posture_baseline([
            feature(head=.2 + (index % 3 - 1) * .005) for index in range(25)
        ])

    def test_calibrated_pose_is_healthy(self):
        result = assess_posture(feature(), self.baseline)
        self.assertTrue(result.visible)
        self.assertEqual(result.reasons, ())

    def test_relative_deviations_are_explainable(self):
        result = assess_posture(
            feature(head=.5, slope=.2, torso=15, width=.45), self.baseline,
        )
        self.assertEqual(
            result.reasons,
            ("head_forward", "shoulder_asymmetry", "torso_lean"),
        )

    def test_low_visibility_is_unknown(self):
        result = assess_posture(feature(confidence=.2), self.baseline)
        self.assertFalse(result.visible)
        self.assertEqual(result.reasons, ())

    def test_calibration_needs_enough_samples(self):
        with self.assertRaises(ValueError):
            build_posture_baseline([feature()] * 5)

    def test_moving_calibration_is_rejected_instead_of_widening_thresholds(self):
        samples = [feature(head=.1 if index % 2 else .5) for index in range(20)]
        with self.assertRaisesRegex(ValueError, "unstable calibration"):
            build_posture_baseline(samples)

    def test_missing_hips_does_not_block_upper_body_baseline(self):
        samples = [feature() for _ in range(20)]
        for sample in samples:
            sample["values"].pop("torso_lean_degrees")
        baseline = build_posture_baseline(samples)
        self.assertNotIn("torso_lean_degrees", baseline.values)
        self.assertEqual(assess_posture(samples[0], baseline).reasons, ())


if __name__ == "__main__":
    unittest.main()
