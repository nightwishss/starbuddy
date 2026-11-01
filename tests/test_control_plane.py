import unittest

from control_plane.registry import ModelRegistry
from control_plane.scheduler import ResourceScheduler, SchedulerError


class ControlPlaneTests(unittest.TestCase):
    def test_registry_has_unique_typed_models(self):
        values = ModelRegistry().list()["models"]
        self.assertGreaterEqual(len(values), 6)
        self.assertEqual(len(values), len({x["id"] for x in values}))
        self.assertIn("qwen3-4b-instruct-2507", {x["id"] for x in values})
        self.assertTrue(all("resources" in x and "backend" in x for x in values))
        self.assertIsNone(next(x for x in values if x["id"] == "qwen3-4b-instruct-2507")["artifacts_ready"])

    def test_leases_enforce_capacity_owner_and_release(self):
        scheduler = ResourceScheduler({"npu_compute": 100}, clock=lambda: 10)
        lease = scheduler.acquire(owner="pose", resources={"npu_compute": 60}, purpose="vision.pose")
        with self.assertRaises(SchedulerError) as caught:
            scheduler.acquire(owner="vision", resources={"npu_compute": 50}, purpose="vision.objects")
        self.assertEqual(caught.exception.code, "resource_busy")
        with self.assertRaises(SchedulerError):
            scheduler.release(lease_id=lease["lease_id"], owner="other")
        self.assertTrue(scheduler.release(lease_id=lease["lease_id"], owner="pose")["released"])

    def test_expired_lease_is_reclaimed(self):
        now = [1.0]
        scheduler = ResourceScheduler({"camera_stream": 1}, clock=lambda: now[0])
        scheduler.acquire(owner="one", resources={"camera_stream": 1}, purpose="capture", ttl_seconds=1)
        now[0] = 2.1
        second = scheduler.acquire(owner="two", resources={"camera_stream": 1}, purpose="capture")
        self.assertEqual(second["owner"], "two")


if __name__ == "__main__": unittest.main()
