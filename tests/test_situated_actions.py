"""Only injected fake TaskClient effects; no services, model, audio or sensors."""

from copy import deepcopy
import json
import threading
import unittest

from gateway.display.situated_actions import ACTION_SESSION, ForegroundBudget, ReminderActionBroker, is_reminder_request
from task_orchestrator.client import TaskClientError, request_task_id
from task_orchestrator.planner import TaskPlanner


class FakeTasks:
    def __init__(self):
        self.created = []
        self.gets = []
        self.cancelled = []
        self.tasks = {}
        self.create_callback = None
        self.create_transform = None
        self.cancel_transform = None
        self.get_transform = None
        self.create_lost_ack = False
        self.cancel_lost_ack = False

    def create(self, utterance, **kwargs):
        self.created.append((utterance, deepcopy(kwargs)))
        task_id = request_task_id(kwargs["source_conversation_id"], kwargs["request_id"])
        value = {"task_id": task_id, "state": "scheduled", "source_conversation_id": kwargs["source_conversation_id"]}
        self.tasks[task_id] = value
        if self.create_callback:
            self.create_callback()
        if self.create_lost_ack:
            raise TimeoutError("synthetic lost create acknowledgement")
        return self.create_transform(deepcopy(value)) if self.create_transform else deepcopy(value)

    def get(self, task_id):
        self.gets.append(task_id)
        if task_id not in self.tasks:
            raise TaskClientError("not_found", "not found")
        value = deepcopy(self.tasks[task_id])
        return self.get_transform(value) if self.get_transform else value

    def cancel(self, task_id):
        self.cancelled.append(task_id)
        value = self.tasks[task_id]
        if value["state"] not in {"completed", "failed", "cancelled"}:
            value["state"] = "cancelled"
        if self.cancel_lost_ack:
            raise TimeoutError("synthetic lost cancel acknowledgement")
        return self.cancel_transform(deepcopy(value)) if self.cancel_transform else deepcopy(value)


class ReminderActionBrokerTests(unittest.TestCase):
    def setUp(self):
        self.now = [100.0]
        self.tasks = FakeTasks()
        self.broker = ReminderActionBroker(self.tasks, clock=lambda: self.now[0])

    def draft(self, text="十分钟后提醒我休息"):
        value = self.broker.propose(text, input_type="text")
        self.assertIsNotNone(value)
        return value

    def test_explicit_minute_reminder_is_only_a_public_unconfirmed_draft(self):
        value = self.draft()
        self.assertEqual(value["minutes"], 10)
        self.assertEqual(value["topic"], "rest")
        self.assertEqual(value["status"], "proposed")
        self.assertEqual(value["expires_in_s"], 60)
        self.assertEqual(self.tasks.created, [])
        self.assertEqual(self.broker.status()["draft"], value)
        self.assertIsNone(self.broker.status()["task"])
        self.assertNotIn("utterance", json.dumps(value))

    def test_confirm_creates_exact_canonical_typed_plan_without_model(self):
        draft = self.draft()
        value = self.broker.confirm(draft["token"])
        self.assertEqual(value["status"], "scheduled")
        self.assertTrue(value["accepted"])
        utterance, options = self.tasks.created[0]
        self.assertEqual(utterance, "10分钟后提醒我休息")
        self.assertEqual(options["source_conversation_id"], ACTION_SESSION)
        self.assertEqual(options["timezone"], "Europe/Madrid")
        proposal = options["proposal"]
        self.assertEqual(proposal, {"summary": "10分钟后提醒：该休息了", "schedule": {"kind": "delay", "delay_seconds": 600},
                                    "notification": "该休息了", "steps": []})
        # Validate through the real plan validator using empty injected
        # catalogue/skills objects: steps=[] means neither can be consulted.
        planner = TaskPlanner(None, catalog=object(), skills=object(), clock=lambda: 100)
        self.assertEqual(planner.validate(proposal, utterance=utterance)["schedule"], {"kind": "delay", "run_at": 700.0})
        self.assertEqual(value["task_id"], request_task_id(ACTION_SESSION, options["request_id"]))

    def test_chinese_and_numeric_minutes_are_canonical_and_bounded(self):
        for text, expected in (("一分钟后提醒我喝水", 1), ("两分钟后提醒我休息", 2),
                               ("二十一分钟后提醒我活动一下", 21), ("一百零五分钟后提醒我休息", 105),
                               ("一千四百四十分钟后提醒我休息", 1440), ("1440分钟后提醒我休息", 1440)):
            with self.subTest(text=text):
                self.assertEqual(self.draft(text)["minutes"], expected)
        self.assertEqual(self.tasks.created, [])

    def test_unsupported_negated_conditional_compound_and_private_topics_never_create(self):
        for text in ("提醒我休息", "不要十分钟后提醒我休息", "十分钟后不要提醒我休息", "十分钟后提醒我休息然后关灯",
                     "十分钟后提醒我休息并喝水", "如果我累了十分钟后提醒我休息", "一分钟后提醒我运行脚本",
                     "十分钟后提醒我私密名字的日记", "0分钟后提醒我休息", "1441分钟后提醒我休息",
                     "01分钟后提醒我休息", "一二分钟后提醒我休息", "十十分钟后提醒我休息", "十秒后提醒我休息",
                     "十分钟后提醒我休息，五分钟后提醒我喝水"):
            with self.subTest(text=text):
                self.assertIsNone(self.broker.propose(text, input_type="voice"))
        self.assertEqual(self.tasks.created, [])

    def test_modify_only_unconfirmed_draft_and_revoke_old_token(self):
        original = self.draft()
        modified = self.draft("改成五分钟")
        self.assertEqual(modified["minutes"], 5)
        self.assertEqual(modified["topic"], "rest")
        self.assertNotEqual(original["token"], modified["token"])
        self.assertFalse(self.broker.confirm(original["token"])["accepted"])
        self.assertEqual(self.tasks.created, [])
        self.assertEqual(self.broker.confirm(modified["token"])["status"], "scheduled")
        self.assertEqual(self.tasks.created[0][1]["proposal"]["schedule"]["delay_seconds"], 300)
        self.assertIsNone(self.broker.propose("改成三分钟"))

    def test_unsupported_modification_revokes_old_unconfirmed_token(self):
        original = self.draft()
        self.assertIsNone(self.broker.propose("改成五分钟然后关闭摄像头"))
        self.assertEqual(self.broker.pending()["status"], "cancelled")
        self.assertFalse(self.broker.confirm(original["token"])["accepted"])
        self.assertEqual(self.tasks.created, [])

    def test_expired_or_cancelled_draft_cannot_be_confirmed(self):
        draft = self.draft()
        self.now[0] = 160
        self.assertFalse(self.broker.confirm(draft["token"])["accepted"])
        self.assertEqual(self.broker.pending()["status"], "expired")
        draft = self.draft()
        self.assertTrue(self.broker.cancel(draft["token"])["cancelled"])
        self.assertFalse(self.broker.confirm(draft["token"])["accepted"])
        self.assertEqual(self.tasks.created, [])

    def test_discard_draft_revokes_confirmation_without_any_task_rpc(self):
        draft = self.draft()
        self.assertTrue(self.broker.discard_draft())
        self.assertFalse(self.broker.confirm(draft["token"])["accepted"])
        self.assertFalse(self.broker.discard_draft())
        self.assertEqual(self.tasks.created, [])
        self.assertEqual(self.tasks.cancelled, [])

    def test_navigation_discard_racing_confirm_never_cancels_real_task(self):
        entered, release = threading.Event(), threading.Event()
        self.tasks.create_callback = lambda: (entered.set(), release.wait(2))
        draft = self.draft()
        results = []
        worker = threading.Thread(target=lambda: results.append(self.broker.confirm(draft["token"])))
        worker.start()
        self.assertTrue(entered.wait(1))
        try:
            self.assertFalse(self.broker.discard_draft())
            with self.assertRaises(RuntimeError):
                self.broker.discard_draft(require_resettable=True)
            self.assertEqual(self.tasks.cancelled, [])
        finally:
            release.set()
            worker.join(2)
        self.assertEqual(results[0]["status"], "scheduled")
        self.assertEqual(len(self.tasks.created), 1)
        self.assertEqual(self.tasks.cancelled, [])
        with self.assertRaises(RuntimeError):
            self.broker.discard_draft(require_resettable=True)

    def test_duplicate_and_concurrent_confirm_create_at_most_once(self):
        entered, release = threading.Event(), threading.Event()
        self.tasks.create_callback = lambda: (entered.set(), release.wait(2))
        draft = self.draft()
        results = []
        worker = threading.Thread(target=lambda: results.append(self.broker.confirm(draft["token"])))
        worker.start()
        self.assertTrue(entered.wait(1))
        try:
            self.assertFalse(self.broker.confirm(draft["token"])["accepted"])
        finally:
            release.set()
            worker.join(2)
        self.assertEqual(len(results), 1)
        self.assertFalse(self.broker.confirm(draft["token"])["accepted"])
        self.assertEqual(len(self.tasks.created), 1)

    def test_lost_create_ack_is_unknown_and_never_retried(self):
        self.tasks.create_lost_ack = True
        draft = self.draft()
        self.assertEqual(self.broker.confirm(draft["token"])["status"], "unknown")
        self.assertIsNone(self.broker.status()["task"])
        self.assertEqual(self.broker.confirm(draft["token"])["status"], "unknown")
        self.assertIsNone(self.broker.propose("五分钟后提醒我喝水"))
        self.assertEqual(len(self.tasks.created), 1)
        self.assertEqual(self.broker.refresh()["status"], "scheduled")
        self.assertEqual(len(self.tasks.created), 1)
        self.assertEqual(self.tasks.gets, [next(iter(self.tasks.tasks))])

    def test_not_found_does_not_reopen_unknown_create(self):
        self.tasks.create_lost_ack = True
        draft = self.draft()
        self.broker.confirm(draft["token"])
        self.tasks.tasks.clear()
        self.assertEqual(self.broker.refresh()["status"], "unknown")
        self.assertEqual(self.broker.confirm(draft["token"])["status"], "unknown")
        self.assertEqual(len(self.tasks.created), 1)

    def test_wrong_create_or_get_receipt_never_claims_success(self):
        for transform in (lambda value: {**value, "task_id": "task_foreign"},
                          lambda value: {**value, "source_conversation_id": "private.other"},
                          lambda value: {**value, "state": "invented"}, lambda _value: {}):
            with self.subTest(transform=transform):
                tasks = FakeTasks()
                broker = ReminderActionBroker(tasks, clock=lambda: self.now[0])
                tasks.create_transform = tasks.get_transform = transform
                draft = broker.propose("十分钟后提醒我休息")
                self.assertEqual(broker.confirm(draft["token"])["status"], "unknown")
                self.assertEqual(broker.refresh()["status"], "unknown")
                self.assertFalse(broker.cancel()["cancelled"])
                self.assertEqual(tasks.cancelled, [])

    def test_cancel_created_task_uses_only_owned_id_and_matching_receipt(self):
        draft = self.draft()
        created = self.broker.confirm(draft["token"])
        cancelled = self.broker.cancel()
        self.assertTrue(cancelled["cancelled"])
        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(self.tasks.cancelled, [created["task_id"]])

    def test_repeated_cancel_preserves_confirmed_terminal_truth_without_more_rpc(self):
        draft = self.draft()
        self.broker.confirm(draft["token"])
        self.assertTrue(self.broker.cancel()["cancelled"])
        again = self.broker.cancel()
        self.assertEqual(again["status"], "cancelled")
        self.assertTrue(again["cancelled"])
        self.assertEqual(len(self.tasks.cancelled), 1)
        self.assertFalse(self.broker.discard_draft(require_resettable=True))

    def test_cancel_during_create_cannot_report_success_until_actual_cancel(self):
        entered, release = threading.Event(), threading.Event()
        self.tasks.create_callback = lambda: (entered.set(), release.wait(2))
        draft = self.draft()
        results = []
        worker = threading.Thread(target=lambda: results.append(self.broker.confirm(draft["token"])))
        worker.start()
        self.assertTrue(entered.wait(1))
        try:
            value = self.broker.cancel()
            self.assertEqual(value["status"], "unknown")
            self.assertFalse(value["cancelled"])
            self.assertEqual(self.tasks.cancelled, [])
        finally:
            release.set()
            worker.join(2)
        self.assertTrue(results[0]["cancelled"])
        self.assertEqual(results[0]["status"], "cancelled")
        self.assertEqual(len(self.tasks.created), 1)
        self.assertEqual(len(self.tasks.cancelled), 1)

    def test_readonly_refresh_reconciles_pending_cancel_but_never_dispatches_it(self):
        self.tasks.create_lost_ack = True
        draft = self.draft()
        self.broker.confirm(draft["token"])
        self.assertFalse(self.broker.cancel()["cancelled"])
        self.assertEqual(self.tasks.cancelled, [])
        status = self.broker.refresh()
        self.assertEqual(status["status"], "scheduled")
        self.assertEqual(status["reason"], "cancel_requires_explicit_retry")
        self.assertEqual(self.tasks.cancelled, [])
        for _ in range(5):
            self.broker.status()
            self.now[0] += 2
            self.broker.refresh()
        self.assertEqual(self.tasks.cancelled, [])
        self.assertTrue(self.broker.cancel()["cancelled"])
        self.assertEqual(len(self.tasks.cancelled), 1)

    def test_lost_cancel_ack_stays_unknown_until_bounded_status_refresh(self):
        draft = self.draft()
        self.broker.confirm(draft["token"])
        self.tasks.cancel_lost_ack = True
        value = self.broker.cancel()
        self.assertEqual(value["status"], "unknown")
        self.assertFalse(value["cancelled"])
        self.assertFalse(self.broker.cancel()["cancelled"])
        self.assertEqual(len(self.tasks.cancelled), 1)
        self.assertEqual(self.broker.refresh()["status"], "cancelled")

    def test_wrong_cancel_receipt_is_unknown_not_cancelled(self):
        draft = self.draft()
        self.broker.confirm(draft["token"])
        self.tasks.cancel_transform = lambda value: {**value, "task_id": "task_foreign"}
        self.assertFalse(self.broker.cancel()["cancelled"])
        self.assertEqual(self.broker.status()["status"], "unknown")

    def test_task_already_completed_does_not_become_cancelled(self):
        draft = self.draft()
        created = self.broker.confirm(draft["token"])
        self.tasks.tasks[created["task_id"]]["state"] = "completed"
        result = self.broker.cancel()
        self.assertEqual(result["status"], "completed")
        self.assertFalse(result["cancelled"])

    def test_refresh_is_throttled_and_only_reads_one_owned_task(self):
        draft = self.draft()
        created = self.broker.confirm(draft["token"])
        for _ in range(10):
            self.broker.refresh()
        self.assertEqual(self.tasks.gets, [created["task_id"]])
        self.now[0] += 2
        self.broker.refresh()
        self.assertEqual(self.tasks.gets, [created["task_id"], created["task_id"]])

    def test_none_task_client_never_claims_created(self):
        broker = ReminderActionBroker(None, clock=lambda: self.now[0])
        draft = broker.propose("十分钟后提醒我休息")
        self.assertFalse(broker.status()["available"])
        self.assertEqual(broker.confirm(draft["token"])["status"], "failed")
        self.assertIsNone(broker.status()["task"])

    def test_clock_reset_cannot_revive_old_draft(self):
        draft = self.draft()
        self.now[0] -= 1
        self.assertFalse(self.broker.confirm(draft["token"])["accepted"])
        self.now[0] = 101
        self.assertFalse(self.broker.confirm(draft["token"])["accepted"])
        self.assertEqual(self.tasks.created, [])

    def test_reminder_request_hint_routes_unsupported_wording_to_clarification(self):
        self.assertTrue(is_reminder_request("不要提醒我"))
        self.assertTrue(is_reminder_request("改成明天早上"))
        self.assertFalse(is_reminder_request("我正在看书"))


class ForegroundBudgetTests(unittest.TestCase):
    def setUp(self):
        self.now = [100.0]
        self.budget = ForegroundBudget(clock=lambda: self.now[0], timeout_seconds=15)

    def test_one_model_and_one_inspection_per_action(self):
        ticket = self.budget.begin()
        self.assertTrue(self.budget.current(ticket))
        self.assertTrue(self.budget.claim(ticket, "model"))
        self.assertFalse(self.budget.claim(ticket, "model"))
        self.assertTrue(self.budget.claim(ticket, "inspection"))
        self.assertFalse(self.budget.claim(ticket, "inspection"))
        self.assertEqual(self.budget.status()["model_calls"], 1)
        self.assertEqual(self.budget.status()["inspections"], 1)

    def test_resource_busy_is_nonblocking_and_does_not_consume_budget(self):
        ticket = self.budget.begin(resource_busy=True)
        self.assertFalse(self.budget.claim(ticket, "model", resource_busy=True))
        self.assertEqual(self.budget.status()["model_calls"], 0)
        self.assertTrue(self.budget.claim(ticket, "model", resource_busy=False))

    def test_cancel_revokes_but_keeps_native_slot_until_finish(self):
        ticket = self.budget.begin()
        self.assertTrue(self.budget.claim(ticket, "model"))
        self.assertTrue(self.budget.cancel())
        self.assertFalse(self.budget.current(ticket))
        self.assertFalse(self.budget.claim(ticket, "inspection"))
        self.assertIsNone(self.budget.begin())
        self.assertTrue(self.budget.status()["busy"])
        self.assertTrue(self.budget.finish(ticket))
        self.assertFalse(self.budget.status()["busy"])
        self.assertIsNotNone(self.budget.begin())

    def test_deadline_does_not_claim_native_cancellation_or_admit_overlap(self):
        ticket = self.budget.begin()
        self.now[0] = 115
        self.assertFalse(self.budget.current(ticket))
        self.assertFalse(self.budget.claim(ticket, "model"))
        self.assertIsNone(self.budget.begin())
        self.assertTrue(self.budget.finish(ticket))

    def test_clock_rollback_or_nan_never_revives_authority(self):
        for bad in (99, float("nan")):
            with self.subTest(bad=bad):
                now = [100.0]
                budget = ForegroundBudget(clock=lambda: now[0])
                ticket = budget.begin()
                now[0] = bad
                self.assertFalse(budget.current(ticket))
                now[0] = 101
                self.assertFalse(budget.current(ticket))
                self.assertTrue(budget.finish(ticket))

    def test_forged_and_stale_ticket_cannot_finish_or_claim_new_action(self):
        original = self.budget.begin()
        forged = {**original, "token": "foreign"}
        self.assertFalse(self.budget.claim(forged, "model"))
        self.assertFalse(self.budget.finish(forged))
        self.assertTrue(self.budget.finish(original))
        newer = self.budget.begin()
        self.assertFalse(self.budget.finish(original))
        self.assertTrue(self.budget.current(newer))

    def test_external_generation_is_checked_before_and_after_admission(self):
        current = [True]
        ticket = self.budget.begin(current=lambda: current[0])
        current[0] = False
        self.assertFalse(self.budget.current(ticket))
        self.assertFalse(self.budget.claim(ticket, "model"))
        self.assertEqual(self.budget.status()["model_calls"], 0)
        self.assertTrue(self.budget.finish(ticket))
        self.assertIsNone(self.budget.begin(current=lambda: False))

    def test_concurrent_begin_and_claim_are_single_flight(self):
        tickets = []
        threads = [threading.Thread(target=lambda: tickets.append(self.budget.begin())) for _ in range(10)]
        for worker in threads:
            worker.start()
        for worker in threads:
            worker.join(2)
        live = [ticket for ticket in tickets if ticket is not None]
        self.assertEqual(len(live), 1)
        claims = []
        threads = [threading.Thread(target=lambda: claims.append(self.budget.claim(live[0], "model"))) for _ in range(10)]
        for worker in threads:
            worker.start()
        for worker in threads:
            worker.join(2)
        self.assertEqual(sum(claims), 1)


if __name__ == "__main__":
    unittest.main()
