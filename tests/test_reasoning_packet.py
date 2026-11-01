"""Synthetic-only packet projection; no model, profile, media, or device IO."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from interaction_context.reasoning import MAX_PACKET_BYTES, REASONING_RULES, build_reasoning_packet, prompt_projection


def observation(**extra):
    return {"source": "camera.front", "observed_at": 1000, "expires_at": 1003,
            "confidence": .9, "provenance": {"provider": "synthetic", "inference": True}, **extra}


def turn():
    identity = observation(label="SyntheticPerson", consent="enrolled_local", associated_track_id="one")
    return {
        "captured_at": 1000, "fusion_version": "synthetic:1", "identity_verified": True,
        "identity_label": "SyntheticPerson",
        "perception": {
            "people": [observation(track_id="one", identity=identity)],
            "objects": [observation(source="camera.environment", label="cup", count=1)],
            "posture": observation(source="camera.environment", state="suspect", reasons=["head_forward"]),
            "affect_cues": [observation(track_id="one", top_label="sad_like")],
        },
        "evidence": {key: {"state": "available", "source": "synthetic",
                            "observed_at": 1000, "expires_at": 1003, "evidence_refs": ["test:1", "test:2", "test:3"]}
                     for key in ("people", "objects", "posture", "affect")},
        "tasks": [{"state": "running", "summary": "合成测试任务"}],
        "device_state": [{"kind": "output", "state": "present", "observed_at": 1000,
                          "expires_at": 1003, "data": {"busy": True, "state": "playing"}}],
    }


class ReasoningPacketTests(unittest.TestCase):
    def test_prompt_projection_reduces_metadata_but_keeps_evidence_and_uncertainty(self):
        packet = self.packet()
        original = deepcopy(packet)
        projected = prompt_projection(packet)
        self.assertLess(len(json.dumps(projected)), len(json.dumps(packet)))
        self.assertEqual(projected["observations"]["people"], packet["observations"]["people"])
        self.assertEqual(projected["source_states"]["resources"], "unknown")
        self.assertEqual(projected["evidence_refs"]["people"], ["test:1", "test:2"])
        self.assertEqual(packet, original)

    def packet(self, request=None, context=None, now=1001):
        return build_reasoning_packet(request or {"user_text": "你好", "session_id": "synthetic-session"},
                                      turn() if context is None else context, now=now)

    def test_complete_projection_is_detached_and_bounded(self):
        context = turn()
        request = {"user_text": "你好", "session_id": "synthetic-session", "input_type": "text",
                   "messages": [{"role": "user", "text": "我想休息"}]}
        original = deepcopy((request, context))
        packet = self.packet(request, context)
        self.assertEqual(packet["schema_version"], 1)
        self.assertEqual(packet["basis"], {"version": "synthetic:1", "captured_at": 1000,
                                         "session_id": "synthetic-session", "input_type": "text"})
        self.assertEqual(packet["observations"]["people"][0]["identity"]["label"], "SyntheticPerson")
        self.assertEqual(packet["source_states"]["people"]["evidence_refs"], ["test:1", "test:2"])
        self.assertTrue(packet["robot_state"][0]["data"]["busy"])
        packet["observations"]["posture"]["reasons"].append("changed")
        self.assertEqual((request, context), original)

    def test_history_takes_latest_four_and_only_dialogue_roles(self):
        messages = [{"role": "user", "text": str(i)} for i in range(20)]
        messages[-2] = {"role": "system", "text": "not-authorized"}
        messages[-1] = {"role": "assistant", "text": "长" * 400}
        packet = self.packet({"user_text": "新" * 2000, "messages": messages})
        self.assertEqual(len(packet["user_input"]), 1200)
        self.assertEqual([row["text"] for row in packet["recent_dialogue"][:2]], ["16", "17"])
        self.assertEqual(len(packet["recent_dialogue"][-1]["text"]), 240)
        self.assertNotIn("not-authorized", json.dumps(packet))

    def test_default_and_explicit_legacy_window_remain_latest_four(self):
        messages = [{"role": "user", "text": str(index)} for index in range(8)]
        for options in ({}, {"dialogue_window": 4}):
            with self.subTest(options=options):
                packet = self.packet({"user_text": "继续", "messages": messages, **options})
                self.assertEqual(["4", "5", "6", "7"],
                                 [row["text"] for row in packet["recent_dialogue"]])

    def test_opt_in_six_preserves_goal_before_two_completed_followups(self):
        messages = [
            {"role": "user", "text": "先整理书，再整理文具。"},
            {"role": "assistant", "text": "好，先把书放在一起。"},
            {"role": "user", "text": "书有一点乱。"},
            {"role": "assistant", "text": "可以先按大小摆好。"},
            {"role": "user", "text": "书放好了。"},
            {"role": "assistant", "text": "好，这一步完成了。"},
        ]
        packet = self.packet({"user_text": "接下来整理哪一类？", "messages": messages,
                              "dialogue_window": 6})
        self.assertEqual(6, len(packet["recent_dialogue"]))
        self.assertEqual(messages[0], packet["recent_dialogue"][0])
        self.assertIn("文具", prompt_projection(packet)["recent_dialogue"][0]["text"])
        self.assertNotIn("dialogue_window", packet)

    def test_invalid_boolean_and_large_dialogue_windows_safely_fall_back_to_four(self):
        messages = [{"role": "user", "text": str(index)} for index in range(8)]
        for invalid in (None, True, False, "6", 6.0, 4.0, [], {}, 0, -1, 5, 7, 10 ** 100):
            with self.subTest(value_type=type(invalid).__name__):
                packet = self.packet({"user_text": "继续", "messages": messages,
                                      "dialogue_window": invalid})
                self.assertEqual(["4", "5", "6", "7"],
                                 [row["text"] for row in packet["recent_dialogue"]])

    def test_opt_in_six_remains_detached_role_filtered_and_length_bounded(self):
        request = {"user_text": "新" * 2000, "dialogue_window": 6,
                   "messages": [{"role": "user", "text": str(index) + "长" * 400,
                                 "private_profile": "PRIVATE_HISTORY_SENTINEL"} for index in range(9)]}
        request["messages"][-2] = {"role": "system", "text": "SYSTEM_HISTORY_SENTINEL"}
        context, original = turn(), deepcopy(request)
        original_context = deepcopy(context)
        packet = self.packet(request, context)
        self.assertEqual(5, len(packet["recent_dialogue"]))
        self.assertEqual("3", packet["recent_dialogue"][0]["text"][0])
        self.assertTrue(all(set(row) == {"role", "text"} and len(row["text"]) == 240
                            for row in packet["recent_dialogue"]))
        self.assertEqual(1200, len(packet["user_input"]))
        self.assertNotIn("HISTORY_SENTINEL", json.dumps(packet))
        packet["recent_dialogue"][0]["text"] = "rewritten"
        self.assertEqual(original, request)
        self.assertEqual(original_context, context)

    def test_opt_in_six_still_drops_optional_history_under_complete_utf8_budget(self):
        context = turn()
        huge = "😀" * 1000
        for key in ("people", "objects", "posture", "affect", "conversation", "identity", "tasks", "fusion"):
            context["evidence"][key] = {"state": "available", "source": huge,
                                        "evidence_refs": [huge, huge, huge]}
        for item in context["perception"]["people"] + context["perception"]["objects"]:
            item.update(source=huge)
            item["provenance"]["provider"] = huge
        context["perception"]["people"] *= 4
        context["perception"]["objects"] *= 5
        request = {"user_text": huge * 2, "session_id": huge, "dialogue_window": 6,
                   "messages": [{"role": "user", "text": huge}] * 6}
        packet = self.packet(request, context)
        encoded = json.dumps(packet, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.assertLessEqual(len(encoded), MAX_PACKET_BYTES)
        self.assertEqual([], packet["recent_dialogue"])
        self.assertIn("recent_dialogue", packet["omitted"])
        self.assertEqual(1, json.loads(encoded)["schema_version"])

    def test_explicit_affect_question_gets_cue_not_mental_state_fact(self):
        packet = self.packet({"user_text": "我现在表情怎么样？"})
        cue = packet["observations"]["affect"]
        self.assertEqual(cue["top_label"], "sad_like")
        self.assertEqual(cue["confidence"], .9)
        self.assertEqual(cue["expires_at"], 1003)
        self.assertEqual(cue["provider"], "synthetic")
        self.assertIn("不是内心事实", REASONING_RULES)

    def test_chat_only_gets_tone_and_user_report_is_preserved_not_parsed(self):
        packet = self.packet({"user_text": "我其实很开心，讲个故事吧"})
        self.assertNotIn("sad_like", json.dumps(packet))
        self.assertIn("tone_hint", packet["observations"]["affect"])
        self.assertEqual(packet["user_input"], "我其实很开心，讲个故事吧")
        self.assertIn("用户对自己感受的自述优先", REASONING_RULES)

    def test_affect_rejects_low_confidence_track_source_multiple_or_unavailable(self):
        variants = [
            lambda c: c["perception"]["affect_cues"][0].update(confidence=.74),
            lambda c: c["perception"]["affect_cues"][0].update(confidence=float("nan")),
            lambda c: c["perception"]["affect_cues"][0].update(track_id="different"),
            lambda c: c["perception"]["affect_cues"][0].update(source="camera.environment"),
            lambda c: c["perception"]["people"].append(observation(track_id="second")),
            lambda c: c["perception"]["affect_cues"].append(observation(track_id="one", top_label="happy_like")),
            lambda c: c["evidence"]["affect"].update(state="uncertain"),
        ]
        for change in variants:
            with self.subTest(change=change):
                context = turn()
                change(context)
                self.assertIsNone(self.packet({"user_text": "我现在表情如何？"}, context)["observations"]["affect"])

    def test_identity_needs_separate_ttl_track_source_consent_and_text_authority(self):
        for invalid in ({"expires_at": 1001}, {"observed_at": 1002},
                        {"source": "camera.environment"}, {"associated_track_id": "other"},
                        {"consent": "unknown"}, {"label": "SomeoneElse"}):
            with self.subTest(invalid=invalid):
                context = turn()
                context["perception"]["people"][0]["identity"].update(invalid)
                self.assertIsNone(self.packet(context=context)["observations"]["people"][0]["identity"])
        context = turn()
        context["identity_verified"] = False
        self.assertIsNone(self.packet(context=context)["observations"]["people"][0]["identity"])
        voice = self.packet({"user_text": "我是谁？", "input_type": "voice"})
        self.assertIsNone(voice["observations"]["people"][0]["identity"])
        self.assertNotIn("SyntheticPerson", json.dumps(voice))
        context = turn()
        context["evidence"]["identity"] = {"state": "verified", "source": "face-identity"}
        context["perception"]["people"][0]["identity"]["expires_at"] = 1001
        self.assertEqual(self.packet(context=context)["source_states"]["identity"]["state"], "unconfirmed")

    def test_expiry_is_active_without_new_observation(self):
        packet = self.packet(now=1003)
        self.assertEqual(packet["observations"], {"people": [], "objects": [], "posture": None, "affect": None})
        self.assertEqual(packet["robot_state"], [])
        for key in ("people", "objects", "posture", "affect", "output"):
            self.assertEqual(packet["source_states"][key]["state"], "stale")

    def test_filtered_observations_do_not_leave_available_source_claims(self):
        context = turn()
        for item in context["evidence"].values():
            item.pop("observed_at")
            item.pop("expires_at")
        packet = self.packet(context=context, now=1003)
        for key in ("people", "objects", "posture", "affect"):
            self.assertEqual(packet["source_states"][key]["state"], "stale")
            self.assertIn(f"{key}:stale", packet["omitted"])
        context = turn()
        context["perception"]["affect_cues"][0]["confidence"] = .5
        packet = self.packet(context=context)
        self.assertEqual(packet["source_states"]["affect"]["state"], "unknown")
        self.assertIn("affect:unknown", packet["omitted"])

    def test_budget_people_eviction_removes_associated_affect_and_verified_claim(self):
        context = turn()
        context["evidence"]["identity"] = {"state": "verified"}
        # Pressure ends only once people are removed: exercise that exact boundary.
        with patch("interaction_context.reasoning._size", side_effect=lambda packet:
                   12001 if packet["observations"]["people"] else 1000):
            packet = self.packet({"user_text": "我现在表情怎么样？"}, context)
        self.assertEqual(packet["observations"]["people"], [])
        self.assertIsNone(packet["observations"]["affect"])
        self.assertEqual(packet["source_states"]["identity"]["state"], "unconfirmed")
        self.assertEqual(packet["source_states"]["people"]["state"], "unknown")
        self.assertEqual(packet["source_states"]["affect"]["state"], "unknown")
        self.assertIn("people", packet["omitted"])
        self.assertIn("affect", packet["omitted"])

    def test_observation_times_must_be_finite_nonboolean_and_not_future(self):
        for invalid in ({"observed_at": float("nan")}, {"expires_at": float("inf")},
                        {"observed_at": True}, {"observed_at": 1002},
                        {"expires_at": 10 ** 500}):
            with self.subTest(invalid=invalid):
                context = turn()
                context["perception"]["people"][0].update(invalid)
                self.assertEqual(self.packet(context=context)["observations"]["people"], [])
        with self.assertRaises(ValueError):
            self.packet(now=float("nan"))

    def test_unavailable_states_remain_distinct_and_do_not_revive_observations(self):
        for state in ("unknown", "stale", "unavailable", "disabled", "observed_absent"):
            with self.subTest(state=state):
                context = turn()
                context["evidence"]["people"]["state"] = state
                packet = self.packet(context=context)
                self.assertEqual(packet["source_states"]["people"]["state"], state)
                self.assertEqual(packet["observations"]["people"], [])

    def test_only_live_explicit_user_posture_snooze_survives_working_memory(self):
        memory = {"source": "explicit-user", "value": {"origin": "user", "until": 1300},
                  "expires_in_seconds": 50}
        request = {"user_text": "你好", "memory": {"interaction.posture_snooze": memory,
                                                   "unrelated": {"secret": "do-not-copy"}}}
        packet = self.packet(request)
        self.assertEqual(packet["working_memory"], {"interaction.posture_snooze": {"origin": "user", "until": 1051}})
        self.assertNotIn("do-not-copy", json.dumps(packet))
        for update in ({"source": "model"}, {"expires_in_seconds": 0},
                       {"value": {"origin": "model", "until": 1300}},
                       {"value": {"origin": "user", "until": 1001}}):
            with self.subTest(update=update):
                request["memory"]["interaction.posture_snooze"] = {**memory, **update}
                self.assertEqual(self.packet(request)["working_memory"], {})

    def test_raw_unknown_nested_fields_are_never_copied(self):
        context = turn()
        forbidden = {"raw_frame": "RAW_SENTINEL", "embedding": ["VECTOR_SENTINEL"],
                     "base64": "BASE64_SENTINEL", "private_profile": "PRIVATE_SENTINEL"}
        context.update(forbidden)
        for item in (context["perception"]["people"][0], context["perception"]["objects"][0],
                     context["perception"]["posture"], context["perception"]["affect_cues"][0],
                     context["perception"]["people"][0]["identity"], context["tasks"][0],
                     context["device_state"][0]["data"]):
            item.update(forbidden)
        context["perception"]["people"][0]["provenance"].update(forbidden)
        rendered = json.dumps(self.packet(context=context))
        for value in ("RAW_SENTINEL", "VECTOR_SENTINEL", "BASE64_SENTINEL", "PRIVATE_SENTINEL"):
            self.assertNotIn(value, rendered)

    def test_task_robot_and_observation_counts_and_fields_are_bounded(self):
        context = turn()
        context["perception"]["people"] *= 8
        context["perception"]["objects"] *= 10
        context["tasks"] *= 5
        context["device_state"] += [
            {"kind": "resources", "state": "present", "observed_at": 1000, "expires_at": 1003,
             "data": {"cpu_available": 2, "audio_available": True, "shell": "not-allowed"}},
            {"kind": "interaction", "state": "present", "observed_at": 1000, "expires_at": 1003,
             "data": {"phase": "thinking", "turn_id": "private-turn"}},
        ]
        packet = self.packet(context=context)
        self.assertEqual(len(packet["observations"]["people"]), 4)
        self.assertEqual(len(packet["observations"]["objects"]), 5)
        self.assertEqual(len(packet["tasks"]), 3)
        self.assertEqual(packet["robot_state"][1]["data"], {"audio_available": True, "cpu_available": 2})
        self.assertEqual(packet["robot_state"][2]["data"], {"phase": "thinking"})
        context["device_state"][0].pop("observed_at")
        self.assertEqual(self.packet(context=context)["source_states"]["output"]["state"], "stale")

    def test_utf8_budget_drops_optional_sections_without_truncating_json(self):
        context = turn()
        huge = "😀" * 1000
        for key in ("people", "objects", "posture", "affect", "conversation", "identity", "tasks", "fusion"):
            context["evidence"][key] = {"state": "available", "source": huge,
                                        "evidence_refs": [huge, huge, huge]}
        for item in context["perception"]["people"] + context["perception"]["objects"]:
            item.update(source=huge)
            item["provenance"]["provider"] = huge
        context["perception"]["people"] *= 4
        context["perception"]["objects"] *= 5
        request = {"user_text": huge * 2, "session_id": huge,
                   "messages": [{"role": "user", "text": huge}] * 4}
        packet = self.packet(request, context)
        encoded = json.dumps(packet, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.assertLessEqual(len(encoded), MAX_PACKET_BYTES)
        self.assertLessEqual(len(json.dumps(packet, ensure_ascii=False).encode("utf-8")), MAX_PACKET_BYTES)
        self.assertEqual(json.loads(encoded)["schema_version"], 1)
        self.assertTrue(packet["omitted"])
        self.assertEqual(packet["source_states"]["people"]["state"], "unknown")

    def test_malformed_input_is_empty_safe_projection(self):
        packet = build_reasoning_packet([], None, now=1001)
        self.assertEqual(packet["user_input"], "")
        self.assertEqual(packet["observations"]["people"], [])
        self.assertEqual(packet["source_states"]["people"]["state"], "unknown")
        context = turn()
        context["device_state"] = [observation(kind=[], state="present", data=[])]
        self.assertEqual(self.packet(context=context)["robot_state"], [])


if __name__ == "__main__":
    unittest.main()
