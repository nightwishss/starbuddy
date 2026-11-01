import unittest

from task_orchestrator.planner import PlanError
from task_orchestrator.skills import RuntimeSkillExecutor, SkillCatalog


class FakeRuntime:
    def __init__(self): self.calls=[]
    def call(self,*args,**kwargs):
        self.calls.append((args,kwargs)); return {"ok":True,"result":{"accepted":True}}


class TaskSkillTests(unittest.TestCase):
    def test_monitor_posture_defaults_are_host_owned_and_bounded(self):
        step=SkillCatalog().validate_step({
            "skill":"monitor_posture","arguments":{"duration_seconds":1800},"label":"守护坐姿",
        },"接下来半小时帮我看着坐姿")
        self.assertEqual(step["arguments"],{
            "duration_seconds":1800,"bad_duration_seconds":20,"cooldown_seconds":180,
        })
        with self.assertRaises(PlanError):
            SkillCatalog().validate_step({
                "skill":"monitor_posture","arguments":{"duration_seconds":30},"label":"x",
            },"检查坐姿")

    def test_runtime_executor_adds_task_identity_not_model(self):
        runtime=FakeRuntime(); executor=RuntimeSkillExecutor(runtime)
        result=executor.call("monitor_posture",{
            "duration_seconds":600,"bad_duration_seconds":20,"cooldown_seconds":180,
        },task_id="task_"+"a"*32,request_id="request-1")
        self.assertTrue(result["ok"])
        args,options=runtime.calls[0]
        self.assertEqual(args[0],"perception.posture.monitor")
        self.assertEqual(args[1]["task_id"],"task_"+"a"*32)
        self.assertEqual(options,{"timeout_ms":30000,"request_id":"request-1"})
        executor.stop("monitor_posture",task_id="task_"+"a"*32,request_id="request-stop")
        args,options=runtime.calls[1]
        self.assertEqual(args,("perception.posture.stop",{"task_id":"task_"+"a"*32}))
        self.assertEqual(options,{"timeout_ms":3000,"request_id":"request-stop"})


if __name__ == "__main__": unittest.main()
