"""Synthetic reasoning-boundary tests: no model, device, profile or framework I/O."""

import copy
import json
import threading
import unittest
from unittest.mock import patch

from gateway.agent.situated import (GENERIC_CONTEXT_RULES, SYSTEM_PROMPT, TASK_RULES, WORKSPACE_RULES,
                                    SituatedDecisionError, SituatedReasoner, _prepare_packet, parse_situated_decision)
from interaction_context.reasoning import REASONING_RULES, build_reasoning_packet
from interaction_context.situation import SituationState
from interaction_context.workspace import Workspace


def decision(intent="help_task", response="我们先看看题目，原来有几个苹果？", capability=None):
    return json.dumps({"intent": intent, "response": response,
                       "requested_capability": capability}, ensure_ascii=False)


class FakeModel:
    def __init__(self, reply=None, callback=None):
        self.reply = reply or decision()
        self.callback = callback
        self.calls = []

    def propose(self, prompt):
        self.calls.append(prompt)
        if self.callback:
            self.callback()
        return self.reply


class SituatedAgentTests(unittest.TestCase):
    def setUp(self):
        self.now = [100.0]
        self.state = SituationState(clock=lambda: self.now[0])
        self.state.select("math-apples-1")
        self.ticket = self.state.begin("text")
        self.packet = {
            "situation": self.state.packet(self.ticket),
            "reasoning": build_reasoning_packet(
                {"user_text": "这个不会", "session_id": "DO_NOT_SEND_SESSION"},
                {"captured_at": 100, "evidence": {}}, now=100),
            "user_input": "这个不会", "allowed_capabilities": [],
        }

    def run_reasoner(self, model=None, **kwargs):
        reasoner = SituatedReasoner(model or FakeModel(), clock=lambda: self.now[0])
        return reasoner.analyze(self.packet, current=lambda: self.state.current(self.ticket), **kwargs)

    def workspace(self, with_attention=False):
        workspace = Workspace(clock=lambda: self.now[0])
        if with_attention:
            self.packet["reasoning"]["observations"]["objects"] = [
                {"label": "cup", "count": 2, "source": "camera.environment", "observed_at": 99, "expires_at": 130}]
            self.packet["reasoning"]["observations"]["people"] = [
                {"track_id": "anonymous-track", "identity": None, "source": "camera.front", "observed_at": 99, "expires_at": 130}]
            self.packet["reasoning"]["source_states"]["objects"]["state"] = "available"
            self.packet["reasoning"]["source_states"]["people"]["state"] = "available"
        snapshot = workspace.ingest(self.packet["reasoning"])
        if with_attention:
            snapshot = workspace.select_attention(snapshot["objects"][0]["ref"], snapshot["revision"])
        workspace.set_goal("inspect_object" if with_attention else "conversation",
                           "一起看看选中的物体" if with_attention else "一起安排今天的学习", "touch" if with_attention else "text")
        workspace.input("DO_NOT_SEND_RAW_WORKSPACE_HISTORY", "voice")
        workspace.reply("DO_NOT_SEND_RAW_REPLY_HISTORY")
        self.packet["workspace"] = workspace.packet()
        return workspace

    def test_actual_workspace_packet_with_namespaced_refs_and_event_metadata_reaches_model(self):
        workspace = self.workspace(with_attention=True)
        model = FakeModel()
        self.assertEqual(self.run_reasoner(model)["status"], "ok")
        data = json.loads(model.calls[0].split("\n当前宿主情境数据：\n", 1)[1])
        self.assertEqual(data["workspace"], workspace.packet())
        self.assertRegex(data["workspace"]["attention"]["ref"], r"object_[0-9]+_[0-9]+")
        self.assertRegex(data["workspace"]["goal"]["id"], r"goal_[0-9]+_[0-9]+")
        self.assertEqual(data["workspace"]["actors"]["microphone_speaker"],
                         {"identity": "unknown", "profile_authorized": False})
        self.assertTrue(any(event.get("source") == "input.touch" for event in data["workspace"]["recent_events"]))
        self.assertTrue(any(event.get("source") == "input.voice" for event in data["workspace"]["recent_events"]))
        self.assertTrue(any(event.get("source") == "agent.output_proposal" for event in data["workspace"]["recent_events"]))
        self.assertNotIn("DO_NOT_SEND_RAW_WORKSPACE_HISTORY", model.calls[0])
        self.assertNotIn("DO_NOT_SEND_RAW_REPLY_HISTORY", model.calls[0])
        self.assertIn("不证明谁在说话或谁触屏", model.calls[0])
        self.assertIn("不能根据用户原话或goal proposed/awaiting_confirmation/submitting", model.calls[0])

    def test_generic_goal_help_does_not_require_a_quiz_task(self):
        self.state.clear()
        self.ticket = self.state.begin("text")
        self.packet["situation"] = self.state.packet(self.ticket)
        self.workspace()
        result = self.run_reasoner(FakeModel(decision("help_task", "我们可以先确定今天最重要的一件事。")))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["intent"], "help_task")
        # A generic plan still cannot become a quiz answer or an execution.
        self.assertEqual(self.run_reasoner(FakeModel(decision("answer_task", "做完啦。")))["status"], "invalid")

    def test_admitted_dialogue_compacts_only_redundant_receipts_without_mutation(self):
        self.workspace(with_attention=True)
        self.packet["reasoning"]["recent_dialogue"] = [
            {"role": "user", "text": "一起看看这个杯子。"},
            {"role": "assistant", "text": "你想了解它的什么？"}]
        original = copy.deepcopy(self.packet)
        model = FakeModel()
        self.assertEqual(self.run_reasoner(model)["status"], "ok")
        data = json.loads(model.calls[0].split("\n当前宿主情境数据：\n", 1)[1])
        expected = copy.deepcopy(original["workspace"])
        expected["recent_events"] = [event for event in expected["recent_events"]
                                     if event["kind"] not in {"input_received", "reply_recorded"}]
        self.assertLess(len(expected["recent_events"]), len(original["workspace"]["recent_events"]))
        self.assertTrue(any(event["kind"] == "attention_selected" for event in expected["recent_events"]))
        self.assertEqual(data["workspace"], expected)
        self.assertEqual(data["reasoning"], _prepare_packet(original)["reasoning"])
        self.assertEqual(self.packet, original)

    def test_compacted_receipts_still_require_full_provenance_validation(self):
        self.workspace(with_attention=True)
        self.packet["reasoning"]["recent_dialogue"] = [{"role": "user", "text": "看看杯子。"}]
        for kind in ("input_received", "reply_recorded"):
            with self.subTest(kind=kind):
                candidate = copy.deepcopy(self.packet)
                event = next(event for event in candidate["workspace"]["recent_events"] if event["kind"] == kind)
                event["source"] = "audio.capture"
                model = FakeModel()
                result = SituatedReasoner(model).analyze(candidate, current=lambda: True)
                self.assertEqual(result["status"], "invalid")
                self.assertEqual(model.calls, [])

    def test_workspace_extra_private_raw_or_executable_fields_rejected_before_model(self):
        self.workspace(with_attention=True)
        for mutate in (
            lambda value: value.update(profile={"name": "private"}),
            lambda value: value["goal"]["slots"].update(tool="audio.capture"),
            lambda value: value["attention"].update(bbox=[0, 0, 1, 1]),
            lambda value: value["attention"].update(source="/private/media.jpg"),
            lambda value: value["actors"].update(microphone_speaker={"identity": "April", "profile_authorized": True}),
            lambda value: value["actors"]["touch_operator"].update(profile_authorized=0),
            lambda value: value["recent_events"][-1].update(text="private raw utterance"),
            lambda value: value["recent_events"][-1].update(source="audio.capture"),
        ):
            with self.subTest(mutate=mutate):
                candidate = copy.deepcopy(self.packet)
                mutate(candidate["workspace"])
                model = FakeModel()
                result = SituatedReasoner(model).analyze(candidate, current=lambda: True)
                self.assertEqual(result["status"], "invalid")
                self.assertEqual(model.calls, [])

    def test_reminder_workspace_only_accepts_typed_public_slots(self):
        workspace = Workspace(clock=lambda: self.now[0])
        workspace.set_goal("reminder", "稍后休息", "voice", {"minutes": 10, "topic": "rest"})
        self.packet["workspace"] = workspace.packet()
        self.assertEqual(self.run_reasoner()["status"], "ok")
        self.packet["workspace"]["goal"]["slots"]["minutes"] = True
        model = FakeModel()
        self.assertEqual(self.run_reasoner(model)["status"], "invalid")
        self.assertEqual(model.calls, [])

    def test_one_call_grounds_selected_task_and_labels_local_honestly(self):
        model = FakeModel()
        result = self.run_reasoner(model)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["backend"], "local-bounded")
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(len(model.calls), 1)
        self.assertIn("苹果加法", model.calls[0])
        self.assertIn("explicit_ui_selection", model.calls[0])
        self.assertIn("profile_authorized", model.calls[0])
        self.assertNotIn("DO_NOT_SEND_SESSION", model.calls[0])
        self.assertEqual(result["requested_capability"], None)

    def test_propose_preferred_to_ordinary_chat_cleanup(self):
        class Both(FakeModel):
            def chat(self, *_args, **_kwargs):
                raise AssertionError("ordinary chat must not reformat structured JSON")
        self.assertEqual(self.run_reasoner(Both())["status"], "ok")

    def test_situated_proposal_route_is_preferred_without_legacy_or_chat_call(self):
        class Specialized(FakeModel):
            def __init__(self):
                super().__init__()
                self.situated_calls = []
            def propose_situated(self, prompt):
                self.situated_calls.append(prompt)
                return self.reply
            def chat(self, *_args, **_kwargs):
                raise AssertionError("ordinary chat must not run")
        model = Specialized()
        result = self.run_reasoner(model)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(len(model.situated_calls), 1)
        self.assertEqual(model.calls, [])

    def test_situated_only_runtime_is_available_and_still_strictly_validated(self):
        class SituatedOnly:
            def __init__(self, reply):
                self.reply, self.calls = reply, []
            def propose_situated(self, prompt):
                self.calls.append(prompt)
                return self.reply
        for reply, expected in ((decision(), "ok"),
                                (decision("inspect_scene", "看看画面。", "perception.scene"), "invalid"),
                                (decision()[:-1] + ',"confidence":1}', "invalid")):
            with self.subTest(expected=expected, reply=reply):
                model = SituatedOnly(reply)
                result = self.run_reasoner(model)
                self.assertEqual(result["status"], expected)
                self.assertEqual(result["backend"], "local-bounded")
                self.assertEqual(result["model_calls"], 1)
                self.assertEqual(len(model.calls), 1)

    def test_situated_proposal_failure_never_retries_legacy_proposal(self):
        class FailingSpecialized(FakeModel):
            def __init__(self):
                super().__init__()
                self.situated_calls = 0
            def propose_situated(self, _prompt):
                self.situated_calls += 1
                raise OSError("PRIVATE")
        model = FailingSpecialized()
        result = self.run_reasoner(model)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(model.situated_calls, 1)
        self.assertEqual(model.calls, [])
        self.assertNotIn("PRIVATE", json.dumps(result))

    def test_prompt_defines_selected_task_help_with_two_short_quiz_contrasts(self):
        model = FakeModel()
        self.assertEqual(self.run_reasoner(model)["status"], "ok")
        prompt = model.calls[0]
        self.assertIn("help_task=求提示或解释", prompt)
        self.assertIn("answer_task=提交选题答案", prompt)
        self.assertIn("clarify=指代或目标不明", prompt)
        self.assertIn("不因“不会/为什么”而clarify", prompt)
        self.assertIn("“为什么”→help_task", prompt)
        self.assertIn("“答案是七”→answer_task", prompt)
        self.assertEqual(prompt.count("→"), 2)
        self.assertNotIn("没有选题，用户“这个不会”", prompt)
        self.assertIn("限80字", prompt)

    def test_nonquiz_prompt_uses_context_without_quiz_examples_or_dropping_boundaries(self):
        self.state.clear()
        self.ticket = self.state.begin("text")
        self.packet["situation"] = self.state.packet(self.ticket)
        self.workspace(with_attention=True)
        self.packet["reasoning"]["recent_dialogue"] = [
            {"role": "user", "text": "先收好文具，再准备运动服。"},
            {"role": "assistant", "text": "好的，按这个顺序。"}]
        self.packet["reasoning"]["source_states"]["posture"]["state"] = "stale"
        model = FakeModel(decision("chat", "先收好文具。"))
        self.assertEqual(self.run_reasoner(model)["status"], "ok")
        prompt = model.calls[0]
        instructions, serialized = prompt.split("\n当前宿主情境数据：\n", 1)
        self.assertNotIn("短对照", instructions)
        self.assertNotIn("→", instructions)
        self.assertNotIn("你想一起看哪道题", instructions)
        for rule in ("recent_dialogue", "遵循最新目标/顺序", "不用先选题",
                     "聚合物体类别", "不编当前实物的材质、方位、持有人或内容",
                     "未知说话者保持未知", "缺失、过期或关闭的来源不能解释为正常或无人",
                     "不能根据用户原话或goal proposed/awaiting_confirmation/submitting"):
            self.assertIn(rule, instructions)
        self.assertNotIn("answer_task", instructions)
        self.assertIn("物体常识/顺序跟进用chat", instructions)
        data = json.loads(serialized)
        self.assertEqual(data["reasoning"]["recent_dialogue"], self.packet["reasoning"]["recent_dialogue"])
        self.assertEqual(data["reasoning"]["source_states"]["posture"], "stale")
        self.assertEqual(data["workspace"], {**self.packet["workspace"], "recent_events": [
            event for event in self.packet["workspace"]["recent_events"]
            if event["kind"] not in {"input_received", "reply_recorded"}]})
        self.assertEqual(data["allowed_capabilities"], [])
        self.assertNotIn("DO_NOT_SEND_SESSION", prompt)
        self.assertNotIn("DO_NOT_SEND_RAW_WORKSPACE_HISTORY", prompt)

    def test_role_subject_rule_keeps_user_state_separate_from_assistant(self):
        self.state.clear()
        self.ticket = self.state.begin("text")
        self.packet["situation"] = self.state.packet(self.ticket)
        self.workspace()
        self.packet["user_input"] = "那我先做哪个？"
        self.packet["reasoning"]["user_input"] = self.packet["user_input"]
        self.packet["reasoning"]["recent_dialogue"] = [
            {"role": "user", "text": "先整理书桌，再阅读。我已准备好了。"},
            {"role": "assistant", "text": "好，按你的顺序来。"},
        ]
        model = FakeModel(decision("chat", "你先整理书桌。"))
        result = self.run_reasoner(model)
        self.assertEqual("你先整理书桌。", result["response"])
        instructions, serialized = model.calls[0].split("\n当前宿主情境数据：\n", 1)
        self.assertIn("user是用户，assistant是芽芽", instructions)
        self.assertIn("用户计划用第二人称“你”描述，不用“我”代用户", instructions)
        self.assertIn("不冒认用户状态", instructions)
        self.assertIn("回答具体下一步，不复述问句", instructions)
        self.assertEqual(self.packet["reasoning"]["recent_dialogue"],
                         json.loads(serialized)["reasoning"]["recent_dialogue"])

    def test_nonquiz_object_answer_is_chat_but_wrong_model_label_stays_rejected(self):
        self.state.clear()
        self.ticket = self.state.begin("text")
        self.packet["situation"] = self.state.packet(self.ticket)
        self.workspace(with_attention=True)
        self.packet["user_input"] = "那它一般适合用来装水吗？"
        self.packet["reasoning"]["user_input"] = self.packet["user_input"]
        good = FakeModel(decision("chat", "一般水杯可以装水，具体材质和状态还不能确认。"))
        self.assertEqual("ok", self.run_reasoner(good)["status"])
        instructions = good.calls[0].split("\n当前宿主情境数据：\n", 1)[0]
        self.assertNotIn("answer_task", instructions)
        self.assertIn('{"intent":"chat","response":"简短中文","requested_capability":null}', instructions)
        self.assertNotIn("help_task|answer_task", instructions)
        wrong = FakeModel(decision("answer_task", "适合装水。"))
        self.assertEqual("invalid", self.run_reasoner(wrong)["status"])
        self.assertEqual(1, len(wrong.calls))

    def test_compressed_prompt_preserves_all_reasoning_rules_and_authority_boundaries(self):
        # Compare against the preceding release's measured static prompt bytes,
        # not total input bytes or a claim about real-model latency.
        self.assertLessEqual(len((SYSTEM_PROMPT + TASK_RULES + WORKSPACE_RULES).encode('utf-8')), 3010 * .8)
        # Keep >=18% static reduction while retaining the explicit role and
        # concrete-next-step rules added after real-model semantic review.
        self.assertLessEqual(len((SYSTEM_PROMPT + GENERIC_CONTEXT_RULES + WORKSPACE_RULES).encode('utf-8')), 2811 * .82)
        self.assertIn(REASONING_RULES, SYSTEM_PROMPT)
        for text in ("不读记忆/个人资料", "不调用工具/录音/开相机/改变设备任务", "不声称已执行保存",
                     "仅allowed_capabilities包含perception.scene且intent=inspect_scene", "是否执行由宿主判定"):
            self.assertIn(text, SYSTEM_PROMPT)
        for text in ("profile_authorized=false", "不证明谁在说话或谁触屏", "只认宿主回执",
                     "scheduled不等于执行/实际听到", "禁输出工具名/参数"):
            self.assertIn(text, WORKSPACE_RULES)

    def test_prompt_json_is_compact_without_changing_projected_data(self):
        model = FakeModel()
        self.assertEqual(self.run_reasoner(model)["status"], "ok")
        serialized = model.calls[0].split("\n当前宿主情境数据：\n", 1)[1]
        data = json.loads(serialized)
        self.assertEqual(serialized, json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
        self.assertEqual(data["situation"], self.packet["situation"])
        self.assertEqual(data["user_input"], self.packet["user_input"])

    def test_hint_request_minimizes_answer_key_without_mutating_input(self):
        self.packet["user_input"] = "这个我不会，能先给一点提示吗？"
        self.packet["reasoning"]["user_input"] = self.packet["user_input"]
        self.packet["situation"]["task"]["answer"] = "SECRET_ANSWER_KEY_NOT_FOR_HINT"
        self.packet["situation"]["task"]["steps"] = ["SECRET_WORKED_SOLUTION_NOT_FOR_HINT"]
        original = copy.deepcopy(self.packet)
        model = FakeModel()
        self.assertEqual(self.run_reasoner(model)["status"], "ok")
        data = json.loads(model.calls[0].split("\n当前宿主情境数据：\n", 1)[1])
        self.assertEqual(data["response_scope"], "hint_only")
        self.assertNotIn("answer", data["situation"]["task"])
        self.assertNotIn("steps", data["situation"]["task"])
        self.assertEqual(data["situation"]["task"]["hint"], original["situation"]["task"]["hint"])
        self.assertNotIn("SECRET_ANSWER_KEY_NOT_FOR_HINT", model.calls[0])
        self.assertNotIn("SECRET_WORKED_SOLUTION_NOT_FOR_HINT", model.calls[0])
        self.assertEqual(self.packet, original)

    def test_explanation_request_retains_grounding_without_host_intent_rewrite(self):
        self.packet["user_input"] = "为什么下一个是这个数？"
        self.packet["reasoning"]["user_input"] = self.packet["user_input"]
        # Deliberately wrong model label: the host may reject poor UX, but the
        # reasoner must not conceal a classifier failure by rewriting it.
        model = FakeModel(decision("clarify", "你想问哪一步？"))
        result = self.run_reasoner(model)
        self.assertEqual(result["intent"], "clarify")
        data = json.loads(model.calls[0].split("\n当前宿主情境数据：\n", 1)[1])
        self.assertEqual(data["response_scope"], "brief")
        self.assertEqual(data["situation"]["task"]["steps"], self.packet["situation"]["task"]["steps"])

    def test_hint_scope_is_content_limit_not_a_rule_based_intent(self):
        for message, expected in (("给点提示，不要直接告诉我答案", "hint_only"),
                                  ("不用提示，直接告诉我答案", "brief")):
            with self.subTest(message=message):
                self.packet["user_input"] = message
                self.packet["reasoning"]["user_input"] = message
                model = FakeModel(decision("chat", "好的。"))
                result = self.run_reasoner(model)
                self.assertEqual(result["intent"], "chat")
                data = json.loads(model.calls[0].split("\n当前宿主情境数据：\n", 1)[1])
                self.assertEqual(data["response_scope"], expected)

    def test_explicit_chat_only_fallback(self):
        class ChatOnly:
            def __init__(self):
                self.options = None
            def chat(self, prompt, **options):
                self.options = options
                return decision()
        model = ChatOnly()
        self.assertEqual(self.run_reasoner(model)["status"], "ok")
        self.assertEqual(model.options, {"max_completion_tokens": 384, "temperature": .1})

    def test_runtimeclient_is_not_an_invented_llm_capability(self):
        class RuntimeOnly:
            def call(self, *_args, **_kwargs):
                raise AssertionError("no implicit Runtime capability")
        result = self.run_reasoner(RuntimeOnly())
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["backend"], "unavailable")
        self.assertEqual(result["model_calls"], 0)

    def test_default_readiness_reports_framework_blocker_without_claiming_window_size(self):
        state = SituatedReasoner(FakeModel()).status()
        self.assertEqual(state["framework"], "hermes")
        self.assertFalse(state["framework_enabled"])
        self.assertEqual(state["framework_blocker"], "context_window_unverified_and_private_bootstrap")
        self.assertEqual(state["backend"], "local-bounded")
        self.assertFalse(state["cloud_fallback"])

    def test_no_automatic_hermes_import_or_filesystem_access(self):
        with patch("builtins.open", side_effect=AssertionError("unexpected file access")):
            result = self.run_reasoner()
        self.assertEqual(result["status"], "ok")

    def test_cancelled_before_call_costs_no_inference(self):
        model = FakeModel()
        self.state.invalidate()
        result = self.run_reasoner(model)
        self.assertEqual(result["status"], "stale")
        self.assertEqual(model.calls, [])

    def test_task_change_during_inference_discards_output(self):
        result = self.run_reasoner(FakeModel(callback=lambda: self.state.select("pattern-even-1")))
        self.assertEqual(result["status"], "stale")
        self.assertEqual(result["response"], "")

    def test_current_callback_failure_fails_closed(self):
        def failed():
            raise RuntimeError("PRIVATE")
        model = FakeModel()
        result = SituatedReasoner(model).analyze(self.packet, current=failed)
        self.assertEqual(result["status"], "stale")
        self.assertEqual(model.calls, [])
        self.assertNotIn("PRIVATE", json.dumps(result))

    def test_expired_response_is_discarded_without_second_call(self):
        model = FakeModel(callback=lambda: self.now.__setitem__(0, 116.0))
        result = self.run_reasoner(model, timeout_seconds=15)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["response"], "")
        self.assertEqual(len(model.calls), 1)

    def test_invalid_deadline_never_calls(self):
        for timeout in (True, 0, -1, float("nan"), float("inf"), 61, "15"):
            with self.subTest(timeout=timeout):
                model = FakeModel()
                self.assertEqual(self.run_reasoner(model, timeout_seconds=timeout)["status"], "invalid")
                self.assertEqual(model.calls, [])

    def test_backend_exception_does_not_leak_or_retry(self):
        def fail():
            raise OSError("PRIVATE prompt or profile")
        model = FakeModel(callback=fail)
        result = self.run_reasoner(model)
        self.assertEqual(result["status"], "unavailable")
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertEqual(len(model.calls), 1)

    def test_malformed_decision_is_not_repaired_or_retried(self):
        for reply in ('{"intent":"help_task"', "```json\n" + decision() + "\n```", decision() + " extra"):
            with self.subTest(reply=reply):
                model = FakeModel(reply)
                self.assertEqual(self.run_reasoner(model)["status"], "invalid")
                self.assertEqual(len(model.calls), 1)

    def test_extra_confidence_identity_or_tool_fields_rejected(self):
        for key, value in (("confidence", 1), ("identity", "April"),
                           ("tool_calls", [{"name": "audio.capture"}]), ("action", "execute")):
            with self.subTest(key=key):
                parsed = json.loads(decision())
                parsed[key] = value
                self.assertEqual(self.run_reasoner(FakeModel(json.dumps(parsed)))["status"], "invalid")

    def test_capability_is_host_allowlisted_proposal_not_call(self):
        self.packet["allowed_capabilities"] = ["perception.scene", "audio.capture", "system.shutdown"]
        model = FakeModel(decision("inspect_scene", "需要先确认当前画面。", "perception.scene"))
        result = self.run_reasoner(model)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["requested_capability"], "perception.scene")
        self.assertNotIn("audio.capture", model.calls[0])
        self.assertNotIn("system.shutdown", model.calls[0])

    def test_capability_not_admitted_by_host_is_denied(self):
        model = FakeModel(decision("inspect_scene", "看看画面。", "perception.scene"))
        self.assertEqual(self.run_reasoner(model)["status"], "invalid")

    def test_missing_task_cannot_be_answered_as_known(self):
        self.state.clear()
        self.ticket = self.state.begin("text")
        self.packet["situation"] = self.state.packet(self.ticket)
        self.assertEqual(self.run_reasoner()["status"], "invalid")
        result = self.run_reasoner(FakeModel(decision("clarify", "你想问的是哪道题？")))
        self.assertEqual(result["status"], "ok")

    def test_no_arbitrary_profile_or_raw_snapshot_forwarded(self):
        self.packet["profile"] = {"secret": "DO_NOT_SEND_PROFILE"}
        model = FakeModel()
        self.assertEqual(self.run_reasoner(model)["status"], "ok")
        self.assertNotIn("DO_NOT_SEND_PROFILE", model.calls[0])
        self.packet["reasoning"]["observations"]["profile"] = {"name": "private"}
        model = FakeModel()
        self.assertEqual(self.run_reasoner(model)["status"], "invalid")
        self.assertEqual(model.calls, [])

    def test_mismatched_turn_or_unbounded_input_never_calls(self):
        for mutation in (
            lambda packet: packet.update(user_input="另一轮输入"),
            lambda packet: packet["situation"].update(profile={"private": True}),
            lambda packet: packet.update(user_input="x" * 1201),
            lambda packet: packet["reasoning"].update(raw_audio=b"private"),
            lambda packet: packet["reasoning"].update(extra=float("nan")),
        ):
            with self.subTest(mutation=mutation):
                candidate = copy.deepcopy(self.packet)
                mutation(candidate)
                model = FakeModel()
                result = SituatedReasoner(model).analyze(candidate, current=lambda: True)
                self.assertEqual(result["status"], "invalid")
                self.assertEqual(model.calls, [])

    def test_singleflight_keeps_ownership_until_native_call_returns(self):
        entered, release = threading.Event(), threading.Event()
        model = FakeModel(callback=lambda: (entered.set(), release.wait(2)))
        reasoner = SituatedReasoner(model)
        results = []
        worker = threading.Thread(target=lambda: results.append(reasoner.analyze(self.packet, current=lambda: True)))
        worker.start()
        self.assertTrue(entered.wait(1))
        try:
            self.assertTrue(reasoner.status()["busy"])
            result = reasoner.analyze(self.packet, current=lambda: True)
            self.assertEqual(result["status"], "busy")
            self.assertEqual(len(model.calls), 1)
        finally:
            release.set()
            worker.join(2)
        self.assertEqual(len(results), 1)
        self.assertFalse(reasoner.status()["busy"])

    def test_injected_hermes_path_uses_exact_safe_parameters_and_context(self):
        calls = []
        class Agent:
            def run_conversation(self, **kwargs):
                calls.append(kwargs)
                return {"final_response": decision()}
        options = []
        def factory(**kwargs):
            options.append(kwargs)
            return Agent()
        class Specialized(FakeModel):
            def propose_situated(self, _prompt):
                raise AssertionError("injected Hermes takes precedence")
        model = Specialized()
        result = SituatedReasoner(model, hermes_factory=factory).analyze(self.packet, current=lambda: True)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["backend"], "hermes-isolated")
        self.assertEqual(model.calls, [])
        self.assertEqual(options[0]["enabled_toolsets"], [])
        self.assertTrue(options[0]["skip_memory"])
        self.assertTrue(options[0]["skip_context_files"])
        self.assertFalse(options[0]["save_trajectories"])
        self.assertFalse(options[0]["load_soul_identity"])
        self.assertIsNone(options[0]["fallback_model"])
        self.assertEqual(calls[0]["conversation_history"], [])
        self.assertIn("苹果加法", calls[0]["user_message"])
        self.assertEqual(calls[0]["user_message"], json.dumps(
            json.loads(calls[0]["user_message"]), ensure_ascii=False, separators=(",", ":")))

    def test_failed_hermes_never_implicitly_falls_back(self):
        def broken(**_kwargs):
            raise RuntimeError("PRIVATE")
        model = FakeModel()
        result = SituatedReasoner(model, hermes_factory=broken).analyze(self.packet, current=lambda: True)
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(model.calls, [])
        self.assertNotIn("PRIVATE", json.dumps(result))


class SituatedDecisionParserTests(unittest.TestCase):
    def test_duplicate_keys_nonfinite_nested_values_and_wrong_types_rejected(self):
        samples = (
            '{"intent":"chat","intent":"help_task","response":"a","requested_capability":null}',
            '{"intent":"chat","response":"a","requested_capability":NaN}',
            '{"intent":[],"response":"a","requested_capability":null}',
            '{"intent":"chat","response":[],"requested_capability":null}',
            '{"intent":"chat","response":"a","requested_capability":{}}',
            '{"intent":"chat","response":"a","requested_capability":"audio.capture"}',
            decision("chat", "a", "perception.scene"),
            decision("inspect_scene", "a", None),
            decision("chat", "x" * 501),
            decision("chat", "a\u0000b"),
        )
        for sample in samples:
            with self.subTest(sample=sample):
                with self.assertRaises(SituatedDecisionError):
                    parse_situated_decision(sample, allowed_capabilities=["perception.scene"])


if __name__ == "__main__":
    unittest.main()
