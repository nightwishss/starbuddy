import json
import tempfile
import threading
import unittest
from pathlib import Path

from task_orchestrator.engine import TaskOrchestrator, TaskOrchestratorError
from task_orchestrator.planner import PlanError, TaskPlanner
from task_orchestrator.service import TaskService
from task_orchestrator.store import TaskStore


class FakeModel:
    def __init__(self, plan):
        self.plan = plan
        self.prompts = []

    def chat(self, prompt, **options):
        self.prompts.append((prompt, options))
        return json.dumps(self.plan, ensure_ascii=False)


class SequenceModel:
    def __init__(self, values):
        self.values = list(values)
        self.prompts = []

    def chat(self, prompt, **options):
        self.prompts.append((prompt, options))
        return json.dumps(self.values.pop(0), ensure_ascii=False)


class FakeRuntime:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = list(responses or [])

    def call(self, capability, arguments, **options):
        self.calls.append((capability, arguments, options))
        if self.responses:
            return self.responses.pop(0)
        return {"ok": True, "result": {"capability": capability}}


class FakeEvents:
    def __init__(self):
        self.values = []

    def publish(self, topic, source, payload, **options):
        self.values.append((topic, source, payload, options))
        return {"revision": len(self.values)}


class FakeSkills:
    def __init__(self): self.calls=[]; self.stop_calls=[]
    def call(self,name,arguments,**options):
        self.calls.append((name,arguments,options)); return {"ok":True,"result":{"accepted":True}}
    def stop(self,name,**options):
        self.stop_calls.append((name,options)); return {"ok":True,"result":{"stopped":True}}


def plan(*, schedule=None, steps=None, notification="任务完成了"):
    return {
        "summary": "测试任务",
        "schedule": schedule or {"kind": "now"},
        "notification": notification,
        "steps": steps or [],
    }


class TaskOrchestratorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.clock_value = 1000.0
        self.clock = lambda: self.clock_value

    def tearDown(self):
        self.temp.cleanup()

    def build(self, value, *, runtime=None, events=None, skills=None):
        planner = TaskPlanner(FakeModel(value), clock=self.clock)
        store = TaskStore(Path(self.temp.name) / "tasks.json", clock=self.clock)
        return TaskOrchestrator(
            planner, store, runtime or FakeRuntime(), events or FakeEvents(),
            skills=skills, clock=self.clock,
        )

    def test_delay_plan_is_validated_and_persisted(self):
        value = plan(
            schedule={"kind": "delay", "delay_seconds": 600},
            steps=[{"capability": "matrix.icon", "arguments": {"name": "heart"},
                    "label": "显示爱心"}],
        )
        orchestrator = self.build(value)
        task = orchestrator.create("十分钟后让矩阵显示爱心")
        self.assertEqual(task["run_at"], 1600.0)
        self.assertEqual(task["state"], "scheduled")
        self.assertEqual(task["timezone"], "UTC")
        self.assertNotIn("utterance", task)
        self.assertEqual(TaskStore(Path(self.temp.name) / "tasks.json").get(task["task_id"])["summary"], "测试任务")
        self.assertEqual(orchestrator.planner.catalog.timeout_ms("matrix.blink"), 30_000)

    def test_command_must_be_grounded_in_original_utterance(self):
        planner = TaskPlanner(FakeModel(plan(steps=[{
            "capability": "led.set_rgb", "arguments": {"red": 1, "green": 2, "blue": 3},
            "label": "改灯色",
        }])), clock=self.clock)
        with self.assertRaisesRegex(PlanError, "not grounded"):
            planner.plan("告诉我系统温度")

    def test_invalid_first_plan_gets_one_strict_repair(self):
        invalid = plan(steps=[{
            "capability": "matrix.clear", "arguments": {"unexpected": True}, "label": "清除",
        }])
        valid = plan(steps=[{
            "capability": "matrix.clear", "arguments": {}, "label": "清除",
        }])
        model = SequenceModel([invalid, valid])
        result = TaskPlanner(model, clock=self.clock).plan("现在清除矩阵")
        self.assertEqual(result["steps"][0]["arguments"], {})
        self.assertEqual(len(model.prompts), 2)
        self.assertIn("验证错误", model.prompts[1][0])

    def test_second_invalid_plan_is_rejected(self):
        invalid = plan(steps=[{
            "capability": "matrix.clear", "arguments": {"unexpected": True}, "label": "清除",
        }])
        planner = TaskPlanner(SequenceModel([invalid, invalid]), clock=self.clock)
        with self.assertRaisesRegex(PlanError, "plan repair failed"):
            planner.plan("现在清除矩阵")

    def test_due_task_dispatches_steps_in_order_and_publishes_notification(self):
        runtime, events = FakeRuntime(), FakeEvents()
        value = plan(steps=[
            {"capability": "matrix.icon", "arguments": {"name": "smile"}, "label": "显示笑脸"},
            {"capability": "matrix.clear", "arguments": {}, "label": "清除矩阵"},
        ], notification="笑脸任务完成")
        orchestrator = self.build(value, runtime=runtime, events=events)
        task = orchestrator.create("现在让矩阵显示笑脸，然后清除矩阵")
        result = orchestrator.run_due(now=1000)
        self.assertEqual(result["executed"], [task["task_id"]])
        self.assertEqual([item[0] for item in runtime.calls], ["matrix.icon", "matrix.clear"])
        completed = orchestrator.get(task["task_id"])
        self.assertEqual(completed["state"], "completed")
        self.assertTrue(all(step["state"] == "completed" for step in completed["steps"]))
        topics = [item[0] for item in events.values]
        self.assertEqual(topics[-2:], ["task.completed", "task.notification.requested"])

    def test_notification_only_task_is_supported(self):
        runtime, events = FakeRuntime(), FakeEvents()
        orchestrator = self.build(plan(notification="该喝水了"), runtime=runtime, events=events)
        task = orchestrator.create("现在提醒我喝水")
        orchestrator.run_due(now=1000)
        self.assertEqual(runtime.calls, [])
        self.assertEqual(orchestrator.get(task["task_id"])["state"], "completed")
        self.assertEqual(events.values[-1][2]["notification"], "该喝水了")

    def test_skill_task_is_admitted_without_blocking_scheduler(self):
        skills=FakeSkills()
        value=plan(steps=[{"skill":"monitor_posture","arguments":{
            "duration_seconds":1800},"label":"守护坐姿"}],notification="坐姿守护开始")
        orchestrator=self.build(value,skills=skills)
        task=orchestrator.create("接下来半小时帮我看着坐姿")
        orchestrator.run_due(now=1000)
        self.assertEqual(orchestrator.get(task["task_id"])["state"],"completed")
        self.assertEqual(skills.calls[0][0],"monitor_posture")
        self.assertEqual(skills.calls[0][1]["bad_duration_seconds"],20)
        started=next(value for value in orchestrator.events.values if value[0]=="task.started")
        self.assertEqual(started[2]["task_kind"],"skill")

    def test_safe_prelude_then_skill_executes_in_order(self):
        runtime, skills = FakeRuntime(), FakeSkills()
        value = plan(steps=[
            {"capability": "matrix.icon", "arguments": {"name": "heart"}, "label": "显示爱心"},
            {"skill": "monitor_posture", "arguments": {"duration_seconds": 600},
             "label": "守护坐姿"},
        ])
        orchestrator = self.build(value, runtime=runtime, skills=skills)
        task = orchestrator.create("先让矩阵显示爱心，然后监测坐姿十分钟")
        orchestrator.run_due(now=1000)
        self.assertEqual(runtime.calls[0][0], "matrix.icon")
        self.assertEqual(skills.calls[0][0], "monitor_posture")
        self.assertEqual(orchestrator.get(task["task_id"])["skill_state"], "admitted")
        self.assertEqual(orchestrator.cancel(task["task_id"])["skill_state"], "stopped")

    def test_skill_pipeline_rejects_multiple_or_nonfinal_skills(self):
        skill = {"skill": "monitor_posture", "arguments": {"duration_seconds": 600},
                 "label": "守护坐姿"}
        icon = {"capability": "matrix.icon", "arguments": {"name": "heart"},
                "label": "显示爱心"}
        planner = TaskPlanner(FakeModel({}), clock=self.clock)
        utterance = "先让矩阵显示爱心，然后监测坐姿十分钟"
        for steps in ([skill, icon], [skill, skill], [icon, icon, icon, skill]):
            with self.subTest(steps=steps), self.assertRaisesRegex(PlanError, "skill pipeline"):
                planner.validate(plan(steps=steps), utterance=utterance, now=1000)

    def test_failed_prelude_never_starts_skill(self):
        runtime = FakeRuntime([{"ok": False, "error": {"code": "offline", "message": "offline"}}])
        skills = FakeSkills()
        value = plan(steps=[
            {"capability": "matrix.icon", "arguments": {"name": "heart"}, "label": "显示爱心"},
            {"skill": "monitor_posture", "arguments": {"duration_seconds": 600},
             "label": "守护坐姿"},
        ])
        orchestrator = self.build(value, runtime=runtime, skills=skills)
        task = orchestrator.create("先让矩阵显示爱心，然后监测坐姿十分钟")
        orchestrator.run_due(now=1000)
        self.assertEqual(orchestrator.get(task["task_id"])["state"], "failed")
        self.assertEqual(skills.calls, [])

    def test_completed_skill_can_stop_without_cancelling_other_work(self):
        skills=FakeSkills()
        value=plan(steps=[{"skill":"monitor_posture","arguments":{
            "duration_seconds":1800},"label":"守护坐姿"}])
        orchestrator=self.build(value,skills=skills)
        task=orchestrator.create("接下来半小时帮我看着坐姿")
        orchestrator.run_due(now=1000)
        self.assertEqual(orchestrator.get(task["task_id"])["skill_state"],"admitted")
        stopped=orchestrator.cancel(task["task_id"])
        self.assertEqual(stopped["skill_state"],"stopped")
        self.assertEqual(stopped["state"],"completed")
        self.assertEqual(len(skills.stop_calls),1)
        orchestrator.cancel(task["task_id"])
        self.assertEqual(len(skills.stop_calls),1)
        self.assertIn("task.skill.stopped",[item[0] for item in orchestrator.events.values])

    def test_cancel_during_skill_admission_stops_new_lease(self):
        class BlockingSkills(FakeSkills):
            def __init__(self):
                super().__init__(); self.entered=threading.Event(); self.release=threading.Event()
            def call(self,name,arguments,**options):
                self.entered.set(); self.release.wait(2)
                return super().call(name,arguments,**options)
        skills=BlockingSkills()
        value=plan(steps=[{"skill":"monitor_posture","arguments":{
            "duration_seconds":600},"label":"守护坐姿"}])
        orchestrator=self.build(value,skills=skills)
        task=orchestrator.create("接下来十分钟帮我看着坐姿")
        thread=threading.Thread(target=orchestrator.run_due)
        thread.start()
        self.assertTrue(skills.entered.wait(1))
        self.assertEqual(orchestrator.cancel(task["task_id"])["state"],"cancelling")
        skills.release.set(); thread.join(2)
        self.assertEqual(orchestrator.get(task["task_id"])["state"],"cancelled")
        self.assertEqual(len(skills.stop_calls),1)

    def test_runtime_failure_stops_remaining_steps(self):
        runtime = FakeRuntime([{"ok": False, "error": {"code": "HARDWARE_ERROR", "message": "offline"}}])
        value = plan(steps=[
            {"capability": "matrix.icon", "arguments": {"name": "heart"}, "label": "爱心"},
            {"capability": "matrix.clear", "arguments": {}, "label": "清除"},
        ])
        orchestrator = self.build(value, runtime=runtime)
        task = orchestrator.create("现在让矩阵显示爱心再清除")
        orchestrator.run_due(now=1000)
        failed = orchestrator.get(task["task_id"])
        self.assertEqual(failed["state"], "failed")
        self.assertEqual(failed["error"]["code"], "HARDWARE_ERROR")
        self.assertEqual(len(runtime.calls), 1)

    def test_scheduled_task_can_be_cancelled(self):
        orchestrator = self.build(plan(schedule={"kind": "delay", "delay_seconds": 30}))
        task = orchestrator.create("三十秒后提醒我休息")
        cancelled = orchestrator.cancel(task["task_id"])
        self.assertEqual(cancelled["state"], "cancelled")
        self.assertEqual(orchestrator.run_due(now=2000)["executed"], [])

    def test_restart_marks_inflight_task_failed_instead_of_replaying(self):
        path = Path(self.temp.name) / "tasks.json"
        store = TaskStore(path, clock=self.clock)
        store.create({"task_id": "task_test", "state": "running", "created_at": 900,
                      "run_at": 900, "summary": "x", "steps": []})
        recovered = TaskStore(path, clock=self.clock).get("task_test")
        self.assertEqual(recovered["state"], "failed")
        self.assertEqual(recovered["error"]["code"], "interrupted")

    def test_service_rejects_unknown_operation_shape(self):
        service = TaskService(self.build(plan()))
        with self.assertRaisesRegex(TaskOrchestratorError, "invalid task operation"):
            service.dispatch({"operation": "task.create", "utterance": "提醒", "extra": True})
        with self.assertRaisesRegex(TaskOrchestratorError, "invalid task operation"):
            service.dispatch({"operation": "task.run_due", "now": 9999999999})

    def test_service_uses_configured_default_timezone(self):
        orchestrator = self.build(plan())
        service = TaskService(orchestrator, default_timezone="Europe/Madrid")
        task = service.dispatch({"operation": "task.create", "utterance": "现在提醒我"})
        self.assertEqual(task["timezone"], "Europe/Madrid")

    def test_source_conversation_is_persisted_and_propagated_to_events(self):
        events = FakeEvents()
        orchestrator = self.build(plan(), events=events)
        task = orchestrator.create("现在提醒我", source_conversation_id="starbuddy.hdmi.owner")
        self.assertEqual(task["source_conversation_id"], "starbuddy.hdmi.owner")
        self.assertEqual(events.values[0][2]["source_conversation_id"], "starbuddy.hdmi.owner")
        with self.assertRaisesRegex(TaskOrchestratorError, "source conversation"):
            orchestrator.create("现在提醒我", source_conversation_id="../bad session")

    def test_confirmed_proposal_is_deterministically_validated_without_model_rewrite(self):
        model = FakeModel(plan(notification="错误模型文案"))
        planner = TaskPlanner(model, clock=self.clock)
        store = TaskStore(Path(self.temp.name) / "confirmed.json", clock=self.clock)
        orchestrator = TaskOrchestrator(planner, store, FakeRuntime(), FakeEvents(), clock=self.clock)
        proposal = plan(
            schedule={"kind": "delay", "delay_seconds": 60},
            notification="该喝水了",
        )
        task = orchestrator.create("1分钟后提醒我喝水", proposal=proposal)
        self.assertEqual(task["notification"], "该喝水了")
        self.assertEqual(task["run_at"], 1060.0)
        self.assertEqual(model.prompts, [])


if __name__ == "__main__":
    unittest.main()
