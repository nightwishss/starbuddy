import unittest

from gateway.agent.router import ControlPlaneAdmission, ResourceBusy


class FakeControl:
    def __init__(self, fail=False): self.fail=fail; self.released=[]
    def acquire(self, *args, **kwargs):
        if self.fail: raise RuntimeError("busy")
        return {"lease_id":"lease_ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    def release(self, lease_id, owner): self.released.append((lease_id,owner))


class ControlPlaneAdmissionTests(unittest.TestCase):
    def test_releases_cross_process_lease(self):
        client=FakeControl()
        with ControlPlaneAdmission(client).acquire(): pass
        self.assertEqual(client.released[0][1],"phone-gateway")
    def test_busy_is_translated_to_route_error(self):
        with self.assertRaises(ResourceBusy):
            with ControlPlaneAdmission(FakeControl(True)).acquire(): pass

if __name__ == "__main__": unittest.main()
