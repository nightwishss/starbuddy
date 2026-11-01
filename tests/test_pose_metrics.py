#!/usr/bin/env python3

import unittest
from types import SimpleNamespace

from pose_service.metrics import landmark_quality, posture_features


def point(x=.5, y=.5, z=0.0, visibility=.9, presence=.9):
    return SimpleNamespace(x=x, y=y, z=z, visibility=visibility, presence=presence)


class PoseMetricTests(unittest.TestCase):
    def test_upper_body_can_be_valid_without_hips(self):
        landmarks = [point() for _ in range(33)]
        landmarks[23] = point(y=1.2)
        landmarks[24] = point(y=1.2)
        value = landmark_quality(landmarks)
        self.assertTrue(value["upper_body_valid"])
        self.assertFalse(value["full_core_valid"])

    def test_low_visibility_is_not_a_valid_detection(self):
        landmarks = [point() for _ in range(33)]
        landmarks[11] = point(visibility=.2)
        value = landmark_quality(landmarks)
        self.assertFalse(value["upper_body_valid"])

    def test_missing_pose_is_unknown(self):
        self.assertEqual(landmark_quality(None)["upper_body_visibility"], 0.0)

    def test_features_are_normalized_and_hide_coordinates(self):
        landmarks = [point() for _ in range(33)]
        landmarks[7] = point(.45, .25)
        landmarks[8] = point(.55, .25)
        landmarks[11] = point(.35, .5)
        landmarks[12] = point(.65, .5)
        landmarks[23] = point(.4, .9)
        landmarks[24] = point(.6, .9)
        result = posture_features(landmarks)
        self.assertAlmostEqual(result["values"]["head_forward_ratio"], 0.0)
        self.assertAlmostEqual(result["values"]["head_height_ratio"], 0.83333)
        self.assertIn("torso_lean_degrees", result["values"])
        self.assertNotIn("landmarks", result)

    def test_features_require_visible_upper_body(self):
        landmarks = [point() for _ in range(33)]
        landmarks[8] = point(visibility=.1)
        self.assertIsNone(posture_features(landmarks))

    def test_world_landmarks_drive_view_independent_geometry(self):
        image = [point() for _ in range(33)]
        image[7], image[8] = point(.49, .25), point(.51, .25)
        image[11], image[12] = point(.48, .5), point(.52, .5)
        world = [point() for _ in range(33)]
        world[7], world[8] = point(-.02, -.6), point(.02, -.6)
        world[11], world[12] = point(-.2, -.4), point(.2, -.4)
        result = posture_features(image, world)
        self.assertAlmostEqual(result["values"]["head_forward_ratio"], 0.0)
        self.assertAlmostEqual(result["values"]["head_height_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
