#!/usr/bin/env python3
"""Unit tests for the restricted phone gateway."""

import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from gateway.media import MediaError
from gateway.phone.server import (
    GatewayState,
    MAX_MESSAGE_CHARS,
    PAIR_MAX_ATTEMPTS,
    TLS_HANDSHAKE_TIMEOUT_SECONDS,
    TLSThreadingHTTPServer,
    _asset,
    _bounded_starbuddy_history,
    _needs_device_tool,
)
from gateway.agent import RouteResult
from gateway.speech.engine import SpeechError
from governance import DataLabel, PrivacyContext


class PhoneGatewayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.state = GatewayState(
            ["/safe/hermes", "-z"],
            root / "run/pairing-code",
            root / "state/latest-camera.jpg",
            chat_timeout=1,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_pairing_code_is_six_digits_and_one_time(self):
        code = self.state.pairing_code
        self.assertRegex(code, r"^[0-9]{6}$")
        token = self.state.pair(code)
        self.assertTrue(token)
        self.assertTrue(self.state.authenticated(token))
        self.assertIsNone(self.state.pair(code))
        self.assertFalse(self.state.pairing_file.exists())

    def test_invalid_and_expired_sessions_are_rejected(self):
        self.assertFalse(self.state.authenticated("unknown"))
        self.state.sessions["expired"] = time.time() - 1
        conversation = MagicMock(wraps=self.state.conversation)
        self.state.conversation = conversation
        self.assertFalse(self.state.authenticated("expired"))
        self.assertEqual(conversation.reset.call_count, 2)

    def test_pairing_attempts_are_rate_limited(self):
        for _ in range(PAIR_MAX_ATTEMPTS):
            self.assertIsNone(self.state.pair("wrong", "phone"))
        self.assertIsNone(self.state.pair(self.state.pairing_code, "phone"))
        self.assertIsNotNone(self.state.pair(self.state.pairing_code, "different-phone"))

    @patch("gateway.phone.server.subprocess.run")
    def test_chat_uses_argument_vector_without_shell(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, "真实回答\n", "")
        answer, backend = self.state.chat("检查板卡状态")
        self.assertEqual(answer, "真实回答")
        self.assertEqual(backend, "hermes-agent")
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["/safe/hermes", "-z", "检查板卡状态"])
        self.assertNotIn("shell", kwargs)

    def test_local_chat_uses_local_client(self):
        local_llm = MagicMock()
        local_llm.chat.return_value = "本地回答"
        self.state.local_llm = local_llm

        answer, backend = self.state.chat("你好", "local")

        self.assertEqual((answer, backend), ("本地回答", "local-qwen"))
        local_llm.chat.assert_called_once_with("你好")

    def test_local_agent_uses_persistent_client(self):
        client = MagicMock()
        client.chat.return_value = "真实状态"
        self.state.local_agent_client = client
        answer, backend = self.state.chat("检查状态", "local-agent")
        self.assertEqual((answer, backend), ("真实状态", "local-agent"))
        client.chat.assert_called_once_with("检查状态")

    def test_local_agent_requires_configured_client(self):
        with self.assertRaisesRegex(RuntimeError, "尚未配置"):
            self.state.chat("检查状态", "local-agent")

    @patch("gateway.phone.server.subprocess.run")
    def test_chat_failure_is_not_leaked(self, run):
        run.return_value = subprocess.CompletedProcess([], 2, "", "secret detail")
        with self.assertRaisesRegex(RuntimeError, "暂时无法"):
            self.state.chat("hello")

    def test_static_pages_have_no_inline_script(self):
        self.assertNotIn(b"<script>", _asset("pair.html"))
        self.assertNotIn(b"<script>", _asset("index.html"))
        self.assertNotIn(b"<script>", _asset("starbuddy.html"))
        self.assertNotIn(b"<script>", _asset("monitor.html"))

    def test_monitor_assets_and_authenticated_telemetry_contract(self):
        page = _asset("monitor.html")
        script = _asset("monitor.js")
        self.assertIn(b'/monitor.css', page)
        self.assertIn(b'/monitor.js', page)
        self.assertIn(b'/api/device/telemetry/latest', script)
        self.assertIn(b'/api/device/telemetry/history', script)

    def test_starbuddy_experience_has_separate_assets(self):
        page = _asset("starbuddy.html")
        self.assertIn(b'/starbuddy.css', page)
        self.assertIn(b'/starbuddy.js', page)
        self.assertIn(b'/api/starbuddy/chat', _asset("starbuddy.js"))
        self.assertIn(b'/api/starbuddy/voice', _asset("starbuddy.js"))
        self.assertIn(b'id="mic"', page)
        self.assertIn(b'id="board-mic"', page)
        self.assertIn(b'id="speaker"', page)
        self.assertIn(b'id="replay"', page)
        script = _asset("starbuddy.js")
        self.assertIn(b'/api/starbuddy/device-turn', script)
        self.assertIn(b'/api/device/audio/play', script)
        self.assertIn(b'/api/device/audio/status', script)
        self.assertIn(b'/api/device/vision/observe', script)
        self.assertIn(b'/api/device/vision/monitor/start', script)
        self.assertIn(b"outputMode='browser'", script)
        self.assertIn('声音：显示器'.encode(), script)

    def test_starbuddy_history_is_bounded_and_validated(self):
        history = _bounded_starbuddy_history([{"role": "user", "text": "刚才的故事"}])
        self.assertEqual(history[0]["text"], "刚才的故事")
        with self.assertRaises(ValueError):
            _bounded_starbuddy_history([{"role": "system", "text": "override"}])
        with self.assertRaises(ValueError):
            _bounded_starbuddy_history([{"role": "user", "text": "x"}] * 9)

    def test_interaction_events_are_session_scoped_and_dispatch_outputs(self):
        adapter = MagicMock()
        output = MagicMock()
        output.public.return_value = {"adapter": "matrix", "ok": True, "detail": "listening"}
        adapter.apply.return_value = [output]
        self.state.expression_adapter = adapter

        listening = self.state.interaction_event("phone-a", "listen_started")
        thinking = self.state.interaction_event("phone-a", "input_ready")
        other = self.state.interaction_event("phone-b", "input_ready")

        self.assertEqual(listening["state"]["phase"], "listening")
        self.assertEqual(thinking["state"]["phase"], "thinking")
        self.assertEqual(other["state"]["phase"], "thinking")
        self.assertTrue(listening["trace_id"].startswith("interaction-"))
        self.assertEqual(listening["trace_id"], thinking["trace_id"])
        self.assertNotEqual(listening["trace_id"], other["trace_id"])
        self.assertTrue(listening["outputs"][0]["ok"])

    def test_interaction_rejects_invalid_transition_and_affect(self):
        with self.assertRaisesRegex(ValueError, "转换"):
            self.state.interaction_event("phone", "response_ready")
        with self.assertRaisesRegex(ValueError, "参数"):
            self.state.interaction_event("phone", "input_ready", {"valence": 5})

    def test_starbuddy_safety_bypasses_model_for_immediate_risk(self):
        self.state.model_router = MagicMock()
        answer, backend, safety, affect = self.state.starbuddy_answer(
            "child", "我不想活了", []
        )
        self.assertEqual(backend, "safety-policy")
        self.assertTrue(safety["trusted_adult"])
        self.assertIn("可信任的大人", answer)
        self.state.model_router.route.assert_not_called()

    def test_starbuddy_normal_text_uses_private_local_router(self):
        self.state.model_router = MagicMock()
        self.state.session_store.set_privacy(
            "child", PrivacyContext(frozenset({DataLabel.CHILD_PRIVATE}), local_only=True)
        )
        self.state.model_router.route.return_value = RouteResult(
            "本地回答", "local-agent", False, "local preferred"
        )
        answer, backend, safety, affect = self.state.starbuddy_answer("child", "你好", [])
        self.assertEqual((answer, backend), ("本地回答", "local-agent"))
        privacy = self.state.model_router.route.call_args.args[1]
        self.assertIn(DataLabel.CHILD_PRIVATE, privacy.labels)

    def test_product_router_reserves_agent_for_real_device_tools(self):
        self.assertFalse(_needs_device_tool("你好，请介绍一下自己"))
        self.assertFalse(_needs_device_tool("你从摄像头看到了什么"))
        self.assertTrue(_needs_device_tool("检查一下板卡状态和温度"))
        self.assertTrue(_needs_device_tool("点亮矩阵灯"))

    def test_natural_conversation_uses_fast_local_qwen(self):
        self.state.local_llm = MagicMock()
        self.state.local_llm.chat.return_value = "你好呀"
        self.state.model_router = MagicMock()

        answer, backend, _safety, _affect = self.state.starbuddy_answer(
            "child", "你好，请介绍一下自己", []
        )

        self.assertEqual((answer, backend), ("你好呀", "local-qwen"))
        self.state.local_llm.chat.assert_called_once()
        self.state.model_router.route.assert_not_called()

    def test_starbuddy_text_and_voice_share_new_conversation_session(self):
        self.state.local_llm = MagicMock()
        self.state.local_llm.chat.side_effect = ["第一轮回答", "第二轮回答"]
        self.state.model_router = MagicMock()
        owner = "opaque-phone-token"

        first = self.state.run_starbuddy_conversation(owner, "我喜欢机器人", input_type="text")
        second = self.state.run_starbuddy_conversation(owner, "我喜欢什么？", input_type="voice")

        self.assertEqual(first[:2], ("第一轮回答", "local-qwen"))
        self.assertEqual(second[:2], ("第二轮回答", "local-qwen"))
        second_prompt = self.state.local_llm.chat.call_args_list[1].args[0]
        self.assertIn("我喜欢机器人", second_prompt)
        context = self.state.conversation.context(self.state.conversation_id(owner))
        self.assertEqual(len(context["messages"]), 4)
        self.assertEqual(context["messages"][-2]["input_type"], "voice")
        self.assertNotIn(owner, context["session_id"])
        self.assertEqual(self.state.session_store.context(owner), [])

    def test_console_text_and_voice_have_separate_shared_channel(self):
        self.state.local_llm = MagicMock()
        self.state.local_llm.chat.side_effect = ["A", "B"]
        owner = "opaque-phone-token"

        self.state.run_console_conversation(owner, "first", mode="local", input_type="text")
        self.state.run_console_conversation(owner, "second", mode="local", input_type="voice")

        prompt = self.state.local_llm.chat.call_args_list[1].args[0]
        self.assertIn("first", prompt)
        context = self.state.conversation.context(
            self.state.conversation_id(owner, "console")
        )
        self.assertEqual(context["messages"][-2]["input_type"], "voice")
        self.assertNotEqual(
            self.state.conversation_id(owner, "console"),
            self.state.conversation_id(owner, "starbuddy"),
        )

    def test_failed_model_turn_is_closed_without_history(self):
        self.state.local_llm = MagicMock()
        self.state.local_llm.chat.side_effect = RuntimeError("model failed")
        owner = "phone"
        with self.assertRaises(RuntimeError):
            self.state.run_starbuddy_conversation(owner, "你好")
        context = self.state.conversation.context(self.state.conversation_id(owner))
        self.assertIsNone(context["active_turn"])
        self.assertEqual(context["messages"], [])
        self.assertEqual(context["last_turn"]["status"], "failed")

    def test_session_reset_clears_both_conversation_channels(self):
        owner = "phone"
        self.state.local_llm = MagicMock()
        self.state.local_llm.chat.return_value = "回答"
        self.state.run_starbuddy_conversation(owner, "你好")
        self.state.run_console_conversation(owner, "hello", mode="local")

        self.state.reset_conversations(owner)

        self.assertEqual(
            self.state.conversation.context(self.state.conversation_id(owner))["messages"], []
        )
        self.assertEqual(
            self.state.conversation.context(
                self.state.conversation_id(owner, "console"))["messages"], []
        )

    def test_device_request_keeps_bounded_local_agent_route(self):
        self.state.local_llm = MagicMock()
        self.state.model_router = MagicMock()
        self.state.model_router.route.return_value = RouteResult(
            "温度正常", "local-agent", False, "local preferred"
        )

        answer, backend, _safety, _affect = self.state.starbuddy_answer(
            "child", "检查一下板卡状态和温度", []
        )

        self.assertEqual((answer, backend), ("温度正常", "local-agent"))
        self.state.local_llm.chat.assert_not_called()
        self.state.model_router.route.assert_called_once()

    def test_structured_observation_is_trusted_context_not_a_tool(self):
        self.state.model_router = MagicMock()
        self.state.model_router.route.return_value = RouteResult(
            "我看见了", "local-agent", False, "local preferred"
        )
        observation = {"summary": {"counts": {"person": 1, "cup": 2}}}

        self.state.starbuddy_answer("child", "你看到了什么", [], observation)

        prompt = self.state.model_router.route.call_args.args[0]
        self.assertIn("受信任视觉模块", prompt)
        self.assertIn("person×1", prompt)

    def test_starbuddy_can_consume_expiring_perception_working_memory(self):
        self.state.local_llm = MagicMock()
        self.state.local_llm.chat.return_value = "我记得刚才看到了杯子"
        self.state.conversation.remember(
            self.state.conversation_id("child"), "perception.current",
            {"counts": {"cup": 1}, "observed_at": time.time()},
            source="test", ttl_seconds=30,
        )

        self.state.run_starbuddy_conversation("child", "刚才看到了什么？")

        prompt = self.state.local_llm.chat.call_args.args[0]
        self.assertIn("cup×1", prompt)

    def test_message_limit_is_bounded(self):
        self.assertEqual(MAX_MESSAGE_CHARS, 4000)
        self.assertLessEqual(TLS_HANDSHAKE_TIMEOUT_SECONDS, 10)

    @patch("gateway.phone.server.ThreadingHTTPServer.process_request_thread")
    def test_tls_handshake_runs_inside_worker_thread(self, parent):
        server = object.__new__(TLSThreadingHTTPServer)
        server.tls_context = MagicMock()
        raw = MagicMock()
        tls = server.tls_context.wrap_socket.return_value
        address = ("127.0.0.1", 12345)

        server.process_request_thread(raw, address)

        raw.settimeout.assert_called_once_with(TLS_HANDSHAKE_TIMEOUT_SECONDS)
        server.tls_context.wrap_socket.assert_called_once_with(raw, server_side=True)
        tls.settimeout.assert_called_once_with(None)
        parent.assert_called_once_with(tls, address)

    def test_vision_checks_session_ownership_before_runtime(self):
        runtime = MagicMock()
        runtime.call.return_value = {
            "ok": True,
            "result": {"detections": [], "count": 0},
        }
        self.state.runtime_client = runtime
        item = self.state.media_store.create(
            "phone-a", "image/jpeg", b"\xff\xd8pixels\xff\xd9"
        )

        result = self.state.detect_objects(item.media_id, "phone-a", 0.5)
        self.assertEqual(result["count"], 0)
        runtime.call.assert_called_once_with(
            "vision.detect_objects",
            {"media_id": item.media_id, "media_owner": "phone-a", "confidence": 0.5},
            timeout_ms=30_000,
        )
        runtime.reset_mock()
        with self.assertRaisesRegex(MediaError, "无权"):
            self.state.detect_objects(item.media_id, "phone-b", 0.5)
        runtime.call.assert_not_called()

    def test_vision_uses_npu_admission_in_production_path(self):
        runtime = MagicMock()
        runtime.call.return_value = {"ok": True, "result": {"detections": [], "count": 0}}
        admission = MagicMock()
        admission.acquire.return_value.__enter__.return_value = None
        admission.acquire.return_value.__exit__.return_value = False
        self.state.runtime_client = runtime
        self.state.npu_admission = admission
        item = self.state.media_store.create(
            "phone-a", "image/jpeg", b"\xff\xd8pixels\xff\xd9"
        )

        result = self.state.detect_objects(item.media_id, "phone-a", 0.5)

        self.assertEqual(result["count"], 0)
        admission.acquire.assert_called_once_with(timeout=0.1)
        runtime.call.assert_called_once()

    def test_structured_observation_is_local_private_and_remembered(self):
        runtime = MagicMock()
        observed = {
            "observation_id": "observation_ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "observed_at": time.time(),
            "summary": {"counts": {"person": 1}},
            "change": {"changed": False},
        }
        runtime.call.return_value = {"ok": True, "result": observed}
        self.state.runtime_client = runtime

        result = self.state.observe_environment(
            "phone-a", role="camera.front", profile="fast", confidence=0.4
        )

        self.assertEqual(result["summary"]["counts"], {"person": 1})
        self.assertEqual(self.state.latest_observation("phone-a"), observed)
        self.assertTrue(self.state.session_store.privacy("phone-a").local_only)
        memory = self.state.conversation.context(
            self.state.conversation_id("phone-a")
        )["memory"]["perception.current"]
        self.assertEqual(memory["value"]["counts"], {"person": 1})
        self.assertLessEqual(memory["expires_in_seconds"], 30)
        runtime.call.assert_called_once_with(
            "perception.observe",
            {"role": "camera.front", "profile": "fast", "confidence": 0.4,
             "stability_samples": 2},
            timeout_ms=30_000,
        )

    def test_observation_monitor_is_explicit_and_session_scoped(self):
        owner = "phone-a"
        self.state.sessions[owner] = time.time() + 60
        self.state.observe_environment = MagicMock(return_value={
            "observed_at": time.time(), "summary": {"counts": {}},
            "change": {"changed": True},
        })

        started = self.state.start_observation_monitor(owner, interval_seconds=3.0)
        deadline = time.time() + 1
        while self.state.observation_monitor_status(owner).get("samples", 0) < 1:
            self.assertLess(time.time(), deadline)
            time.sleep(0.01)
        with self.assertRaises(RuntimeError):
            self.state.observation_monitor_status("phone-b")
        stopped = self.state.stop_observation_monitor(owner)

        self.assertTrue(started["active"])
        self.assertFalse(stopped["active"])
        self.assertEqual(stopped["status"], "stopped")

    def test_voice_turn_propagates_session_owner_to_stt_and_tts(self):
        owner = "phone-session"
        source = self.state.media_store.create(
            owner, "audio/wav", b"RIFFxxxxWAVEdata"
        )
        generated = self.state.media_store.create(
            owner, "audio/wav", b"RIFFxxxxWAVEdata", kind="generated"
        )
        runtime = MagicMock()
        runtime.call.side_effect = [
            {"ok": True, "result": {"text": "你好"}},
            {"ok": True, "result": {"media_id": generated.media_id}},
        ]
        self.state.runtime_client = runtime

        transcript, answer, audio, backend = self.state.voice_turn(
            source.media_id, owner, answer_provider=lambda text: ("你好呀", "test")
        )

        self.assertEqual((transcript, answer, backend), ("你好", "你好呀", "test"))
        self.assertEqual(audio["media_id"], generated.media_id)
        with self.assertRaises(MediaError):
            self.state.media_store.get(source.media_id, owner)
        self.assertEqual(
            runtime.call.call_args_list[0].args[:2],
            ("speech.transcribe", {"media_id": source.media_id, "media_owner": owner}),
        )
        self.assertEqual(
            runtime.call.call_args_list[1].args[:2],
            ("speech.synthesize", {"text": "你好呀", "media_owner": owner}),
        )

    def test_physical_audio_adapter_uses_fixed_logical_roles(self):
        runtime = MagicMock()
        runtime.call.side_effect = [
            {"ok": True, "result": {"media_id": "audio_ABCDEFGHIJKLMNOPQRSTUVWXYZ"}},
            {"ok": True, "result": {"playback_id": "playback_ABCDEFGHIJKLMNOPQRSTUVWXYZ"}},
            {"ok": True, "result": {"state": "playing"}},
            {"ok": True, "result": {"state": "stopped"}},
        ]
        self.state.runtime_client = runtime

        captured = self.state.capture_board_audio("phone-session", 3000, 80)
        played = self.state.start_board_playback(
            "generated_ABCDEFGHIJKLMNOPQRSTUVWXYZ", "phone-session", 90
        )
        self.state.board_playback_status("phone-session", played["playback_id"])
        self.state.stop_board_playback("phone-session", played["playback_id"])

        self.assertTrue(captured["media_id"].startswith("audio_"))
        self.assertEqual(runtime.call.call_args_list[0].args[:2], (
            "audio.capture_clip", {
                "media_owner": "phone-session", "role": "audio.primary_input",
                "duration_ms": 3000, "sample_rate": 16000, "channels": 1,
                "volume_percent": 80,
            }))
        self.assertEqual(runtime.call.call_args_list[1].args[:2], (
            "audio.start_playback", {
                "media_id": "generated_ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                "media_owner": "phone-session", "role": "audio.primary_output",
                "volume_percent": 90,
            }))

    def test_physical_playback_is_scoped_to_phone_session(self):
        runtime = MagicMock()
        runtime.call.return_value = {
            "ok": True,
            "result": {"playback_id": "playback_ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
        }
        self.state.runtime_client = runtime
        played = self.state.start_board_playback(
            "generated_ABCDEFGHIJKLMNOPQRSTUVWXYZ", "owner-a", 75
        )

        with self.assertRaises(SpeechError):
            self.state.board_playback_status("owner-b", played["playback_id"])

    def test_physical_turn_orchestrates_capture_agent_and_playback(self):
        self.state.capture_board_turn = MagicMock(return_value={
            "detected": True,
            "media_id": "audio_ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "turn": {"stop_reason": "end_silence"},
        })
        self.state.voice_turn = MagicMock(return_value=(
            "你好星伴", "你好呀", {"media_id": "generated_ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
            "local-agent",
        ))
        self.state.start_board_playback = MagicMock(return_value={
            "playback_id": "playback_ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        })

        result = self.state.run_physical_starbuddy_turn("phone-session")

        self.assertTrue(result["detected"])
        self.assertEqual(result["transcript"], "你好星伴")
        self.assertEqual(result["state"]["phase"], "speaking")
        self.state.capture_board_turn.assert_called_once_with("phone-session")
        self.state.start_board_playback.assert_called_once_with(
            "generated_ABCDEFGHIJKLMNOPQRSTUVWXYZ", "phone-session", 100
        )

    @patch("gateway.phone.server.time.sleep")
    def test_physical_capture_interrupts_output_and_waits_for_echo_tail(self, sleep):
        owner = "phone-session"
        playback_id = "playback_ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        runtime = MagicMock()
        runtime.call.return_value = {
            "ok": True, "result": {"playback_id": playback_id, "state": "stopped"},
        }
        self.state.runtime_client = runtime
        self.state.active_playbacks[owner] = playback_id

        policy = self.state.prepare_physical_capture(owner)

        self.assertTrue(policy["interrupted_playback"])
        self.assertEqual(policy["mode"], "half_duplex")
        self.assertFalse(policy["aec_enabled"])
        self.assertGreater(policy["guard_applied_ms"], 0)
        sleep.assert_called_once()
        runtime.call.assert_called_once_with(
            "audio.stop_playback", {"playback_id": playback_id}, timeout_ms=30_000
        )

    def test_physical_turn_returns_without_agent_when_no_speech(self):
        self.state.capture_board_turn = MagicMock(return_value={
            "detected": False, "media_id": None,
            "turn": {"stop_reason": "no_speech"},
        })
        self.state.voice_turn = MagicMock()

        result = self.state.run_physical_starbuddy_turn("phone-session")

        self.assertFalse(result["detected"])
        self.state.voice_turn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
