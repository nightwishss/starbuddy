import unittest

from pose_service.canonical import (
    CanonicalPoint,
    CanonicalPose,
    from_mediapipe,
    from_posenet,
    posture_features,
)


class Point:
    def __init__(self, x, y, visibility=.9, presence=.9, z=0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence


class CanonicalPoseTests(unittest.TestCase):
    def pose(self, score=0.9):
        return CanonicalPose("test", {
            "nose": CanonicalPoint(.50, .15, score),
            "left_ear": CanonicalPoint(.43, .20, score),
            "right_ear": CanonicalPoint(.57, .20, score),
            "left_shoulder": CanonicalPoint(.40, .40, score),
            "right_shoulder": CanonicalPoint(.60, .40, score),
            "left_hip": CanonicalPoint(.42, .70, score),
            "right_hip": CanonicalPoint(.58, .70, score),
        }, 5.0)

    def test_features_preserve_existing_named_semantics(self):
        result = posture_features(self.pose())
        self.assertEqual(result["confidence"], .9)
        self.assertTrue(result["full_core_valid"])
        self.assertAlmostEqual(result["values"]["shoulder_width"], .2)
        self.assertAlmostEqual(result["values"]["head_forward_ratio"], 0.0)
        self.assertAlmostEqual(result["values"]["head_height_ratio"], 1.0)

    def test_missing_or_low_confidence_upper_body_is_rejected(self):
        self.assertIsNone(posture_features(self.pose(score=.09)))

    def test_coordinates_are_bounded(self):
        with self.assertRaises(ValueError):
            CanonicalPoint(1.1, .5, .9)

    def test_adapters_produce_equivalent_features(self):
        landmarks = [Point(.5, .5) for _ in range(25)]
        values = {
            0: (.50, .15), 7: (.43, .20), 8: (.57, .20),
            11: (.40, .40), 12: (.60, .40),
            23: (.42, .70), 24: (.58, .70),
        }
        for index, (x, y) in values.items():
            landmarks[index] = Point(x, y)
        media_pose = from_mediapipe(landmarks, 37.2, "frame-1")
        pose_net = from_posenet({
            name: {"x": point.x, "y": point.y, "score": point.score, "z": point.z}
            for name, point in self.pose().points.items()
        }, 12.5, "frame-1")
        self.assertEqual(posture_features(media_pose), posture_features(pose_net))
        self.assertEqual(media_pose.frame_id, pose_net.frame_id)

    def test_nose_is_part_of_existing_quality_gate(self):
        points = dict(self.pose().points)
        points["nose"] = CanonicalPoint(.5, .15, .09)
        self.assertIsNone(posture_features(CanonicalPose("test", points, 5.0)))

    def test_mediapipe_out_of_frame_point_is_invalid_not_an_exception(self):
        landmarks = [Point(.5, .5) for _ in range(25)]
        landmarks[7] = Point(-.01, .2)
        pose = from_mediapipe(landmarks, 5.0)
        self.assertEqual(pose.points["left_ear"].x, 0.0)
        self.assertEqual(pose.points["left_ear"].score, 0.0)
        self.assertFalse(pose.upper_body_valid(.5))


if __name__ == "__main__":
    unittest.main()
