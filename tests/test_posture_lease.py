import tempfile
import unittest
from pathlib import Path

from posture_service.lease import PostureLeaseError, PostureMonitorLease


class FakePosture:
    def __init__(self,calibrated=True):
        self.calibrated=calibrated; self.enabled=False; self.configurations=[]
    def status(self): return {"calibrated":self.calibrated,"enabled":self.enabled,"state":"healthy"}
    def start(self): self.enabled=True; return self.status()
    def pause(self): self.enabled=False; return self.status()
    def configure_monitor_policy(self,**values): self.configurations.append(values)


class FakeEvents:
    def __init__(self): self.values=[]
    def publish(self,*args,**kwargs): self.values.append((args,kwargs))


class PostureLeaseTests(unittest.TestCase):
    def test_skill_lease_persists_expires_and_pauses(self):
        with tempfile.TemporaryDirectory() as directory:
            now=[1000.0]; posture=FakePosture(); events=FakeEvents()
            lease=PostureMonitorLease(posture,Path(directory)/"lease.json",events=events,
                                      clock=lambda:now[0],autostart=False,start_worker=False)
            result=lease.monitor(task_id="task_"+"a"*32,duration_seconds=600,
                                 bad_duration_seconds=20,cooldown_seconds=180)
            self.assertTrue(result["accepted"]); self.assertTrue(posture.enabled)
            self.assertEqual(result["monitor_session"]["remaining_seconds"],600)
            now[0]=1600
            self.assertTrue(lease.expire_due()); self.assertFalse(posture.enabled)
            self.assertEqual(lease.status()["monitor_mode"],"paused")
            self.assertEqual([value[0][0] for value in events.values],[
                "skill.monitor_posture.started","skill.monitor_posture.completed",
            ])
            lease.close()

    def test_active_lease_is_idempotent_for_same_task_and_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            lease=PostureMonitorLease(FakePosture(),Path(directory)/"lease.json",
                                      clock=lambda:1000,autostart=False,start_worker=False)
            values={"duration_seconds":600,"bad_duration_seconds":20,"cooldown_seconds":180}
            lease.monitor(task_id="task_"+"a"*32,**values)
            self.assertTrue(lease.monitor(task_id="task_"+"a"*32,**values)["idempotent"])
            with self.assertRaisesRegex(PostureLeaseError,"another"):
                lease.monitor(task_id="task_"+"b"*32,**values)

    def test_uncalibrated_posture_is_rejected_without_side_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            posture=FakePosture(False)
            lease=PostureMonitorLease(posture,Path(directory)/"lease.json",
                                      autostart=False,start_worker=False)
            with self.assertRaisesRegex(PostureLeaseError,"calibration"):
                lease.monitor(task_id="task_"+"a"*32,duration_seconds=600)
            self.assertFalse(posture.enabled)

    def test_active_lease_restores_after_service_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"lease.json"; now=[1000.0]
            first=PostureMonitorLease(FakePosture(),path,clock=lambda:now[0],
                                      autostart=False,start_worker=False)
            first.monitor(task_id="task_"+"a"*32,duration_seconds=600)
            restored_posture=FakePosture()
            restored=PostureMonitorLease(restored_posture,path,clock=lambda:1100,
                                         autostart=False,start_worker=False)
            self.assertTrue(restored_posture.enabled)
            self.assertEqual(restored.status()["monitor_session"]["remaining_seconds"],500)

    def test_stop_is_owner_scoped_idempotent_and_persistent(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"lease.json"; events=FakeEvents(); posture=FakePosture()
            lease=PostureMonitorLease(posture,path,events=events,clock=lambda:1000,
                                      autostart=False,start_worker=False)
            owner="task_"+"a"*32
            lease.monitor(task_id=owner,duration_seconds=600)
            with self.assertRaisesRegex(PostureLeaseError,"another"):
                lease.stop(task_id="task_"+"b"*32)
            self.assertTrue(posture.enabled)
            self.assertTrue(lease.stop(task_id=owner)["stopped"])
            self.assertFalse(posture.enabled)
            self.assertFalse(lease.stop(task_id=owner)["stopped"])
            self.assertEqual([value[0][0] for value in events.values].count(
                "skill.monitor_posture.stopped"),1)
            restored=PostureMonitorLease(FakePosture(),path,clock=lambda:1001,
                                         autostart=False,start_worker=False)
            self.assertEqual(restored.status()["monitor_mode"],"paused")


if __name__ == "__main__": unittest.main()
