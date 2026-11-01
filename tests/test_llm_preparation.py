"""Preparation tests use only injected clocks, clients and status/transport fakes."""
import copy
import json
import unittest
from unittest.mock import patch

from gateway.llm.preparation import (
    HEALTH_WAIT_SECONDS, NATIVE_TIMEOUT_SECONDS, STARTUP_QUIET_SECONDS, WARMUP_TEXT,
    MAX_STATUS_BYTES, _http_get, _interaction_probe, _unix_status, main, prepare_once,
)


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []
    def __call__(self):
        return self.now
    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def idle_status():
    return {"lane": {"state": "idle", "lane": None, "persistence_error": None},
            "output": {"ready": True, "current": None, "queued": 0, "output_quarantined": False},
            "interaction": {"availability": "available", "state": {
                "phase": "idle", "user_busy": False, "output_pending": False, "quiet": False}}}


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.state = idle_status()
        self.factory_calls, self.chat_calls, self.probe_calls = [], [], []
        self.reply = "好"
        self.error = None
        self.native_seconds = .2

    def factory(self, **options):
        self.factory_calls.append(options)
        outer = self
        class Client:
            def chat(self, text, **values):
                outer.chat_calls.append((text, values))
                outer.clock.now += outer.native_seconds
                if outer.error:
                    raise outer.error
                return outer.reply
        return Client()

    def probe(self, timeout):
        self.probe_calls.append((self.clock(), timeout))
        return copy.deepcopy(self.state)

    def run_prepare(self, **overrides):
        options = {"client_factory": self.factory, "health_probe": lambda _timeout: True,
                   "idle_probe": self.probe, "clock": self.clock, "sleep": self.clock.sleep}
        options.update(overrides)
        return prepare_once(**options)

    def test_one_background_attempt_has_separate_cold_budget_and_no_output(self):
        with patch("socket.socket", side_effect=AssertionError("no real socket")), \
                patch("builtins.open", side_effect=AssertionError("no file reads")):
            result = self.run_prepare()
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["stage"], "complete")
        self.assertFalse(result["user_response_latency_measured"])
        self.assertEqual(self.factory_calls, [{"timeout": 65.0, "admission_kind": "background"}])
        self.assertEqual(self.chat_calls, [(WARMUP_TEXT, {"max_completion_tokens": 8, "temperature": 0.0})])
        self.assertEqual(len(self.probe_calls), 2)
        self.assertAlmostEqual(result["timing_ms"]["native_warmup"], 200)

    def test_api_health_is_not_itself_model_readiness(self):
        self.error = TimeoutError("PRIVATE_RESPONSE")
        result = self.run_prepare()
        self.assertEqual(result["status"], "failure")
        self.assertFalse(result["ready"])
        self.assertEqual(result["reason"], "native_warmup_failed")
        self.assertEqual(len(self.chat_calls), 1)
        self.assertNotIn("PRIVATE_RESPONSE", json.dumps(result))

    def test_native_failure_never_retries_or_releases_quarantine(self):
        self.error = OSError("PRIVATE")
        result = self.run_prepare()
        self.assertEqual(result["status"], "failure")
        self.assertEqual(len(self.factory_calls), 1)
        self.assertEqual(len(self.chat_calls), 1)

    def test_background_admission_refusal_reports_busy_without_retry(self):
        class AdmissionBusy(RuntimeError):
            code = "llm_foreground_recent"
        self.error = RuntimeError("PRIVATE")
        self.error.__cause__ = AdmissionBusy("PRIVATE")
        result = self.run_prepare()
        self.assertEqual(result["status"], "busy")
        self.assertEqual(result["reason"], "background_admission_busy")
        self.assertEqual(len(self.chat_calls), 1)

    def test_health_wait_is_bounded_and_never_calls_client_when_unavailable(self):
        calls = []
        result = self.run_prepare(health_probe=lambda timeout: calls.append(timeout) or False)
        self.assertEqual(result["reason"], "api_health_unavailable")
        self.assertLessEqual(self.clock() - 100, HEALTH_WAIT_SECONDS)
        self.assertLessEqual(len(calls), 41)
        self.assertEqual(self.factory_calls, [])
        self.assertTrue(all(0 < value <= .5 for value in calls))

    def test_health_poll_retries_only_status_not_inference(self):
        health = iter([ConnectionRefusedError(), False, True])
        def probe(_timeout):
            value = next(health)
            if isinstance(value, Exception):
                raise value
            return value
        result = self.run_prepare(health_probe=probe)
        self.assertTrue(result["ready"])
        self.assertEqual(result["health_checks"], 3)
        self.assertEqual(len(self.chat_calls), 1)

    def test_health_result_arriving_after_deadline_is_not_ready(self):
        def probe(_timeout):
            self.clock.now += 10.1
            return True
        self.assertFalse(self.run_prepare(health_probe=probe)["ready"])
        self.assertEqual(self.factory_calls, [])

    def test_occupied_or_quarantined_status_never_creates_client(self):
        variants = [("lane", "state", "active"), ("lane", "state", "quarantined"),
                    ("lane", "lane", {"owner": "PRIVATE"}), ("lane", "persistence_error", "PRIVATE"),
                    ("output", "ready", False), ("output", "current", {"PRIVATE": "payload"}),
                    ("output", "queued", 1), ("output", "output_quarantined", True),
                    ("output", "tts_quarantined", True)]
        for section, key, value in variants:
            with self.subTest(section=section, key=key):
                self.state = idle_status()
                self.state[section][key] = value
                result = self.run_prepare()
                self.assertEqual(result["status"], "busy")
                self.assertEqual(self.factory_calls, [])
                self.assertNotIn("PRIVATE", json.dumps(result))

    def test_interaction_active_pending_quiet_or_unknown_is_not_idle(self):
        for key, value in (("phase", "thinking"), ("user_busy", True), ("output_pending", True),
                           ("quiet", True), ("user_busy", 0)):
            with self.subTest(key=key, value=value):
                self.state = idle_status()
                self.state["interaction"]["state"][key] = value
                self.assertEqual(self.run_prepare()["status"], "busy")
                self.assertEqual(self.factory_calls, [])

    def test_invalid_or_unavailable_status_fails_closed(self):
        for value in (None, {}, {"lane": {}}, {**idle_status(), "output": {}}):
            with self.subTest(value=value):
                self.assertEqual(self.run_prepare(idle_probe=lambda _timeout: value)["status"], "failure")
        def failure(_timeout):
            raise ValueError("PRIVATE")
        self.assertEqual(self.run_prepare(idle_probe=failure)["status"], "failure")
        self.assertEqual(self.factory_calls, [])

    def test_startup_allows_only_explicit_ui_not_started_reasons(self):
        for reason in ("connection_refused", "not_found"):
            with self.subTest(reason=reason):
                self.setUp()
                self.state["interaction"] = {"availability": "not_started", "reason": reason}
                result = self.run_prepare(startup=True)
                self.assertTrue(result["ready"])
                self.assertAlmostEqual(result["timing_ms"]["startup_quiet_window"], 11000)
                self.assertEqual(len(self.chat_calls), 1)
                self.assertTrue(all(value <= .5 for value in self.clock.sleeps))
        self.setUp()
        self.state["interaction"] = {"availability": "not_started", "reason": "timeout"}
        self.assertEqual(self.run_prepare(startup=True)["status"], "failure")
        self.assertEqual(self.factory_calls, [])

    def test_default_once_requires_ui_even_if_it_has_not_started(self):
        self.state["interaction"] = {"availability": "not_started", "reason": "connection_refused"}
        result = self.run_prepare()
        self.assertEqual(result["status"], "busy")
        self.assertEqual(result["reason"], "interaction_unavailable")
        self.assertEqual(self.factory_calls, [])
        self.assertEqual(self.clock.sleeps, [])

    def test_startup_quiet_window_stops_at_first_activity_without_extension(self):
        def probe(timeout):
            value = self.probe(timeout)
            if self.clock() - 100 >= 1:
                value["interaction"]["state"]["user_busy"] = True
            return value
        result = self.run_prepare(startup=True, idle_probe=probe)
        self.assertEqual(result["status"], "busy")
        self.assertEqual(result["stage"], "startup_quiet_window")
        self.assertAlmostEqual(self.clock() - 100, 1)
        self.assertEqual(self.factory_calls, [])

    def test_startup_ui_appearing_busy_during_quiet_window_is_not_ignored(self):
        def probe(_timeout):
            value = idle_status()
            value["interaction"] = {"availability": "not_started", "reason": "not_found"}
            if self.clock() - 100 >= .5:
                value = idle_status()
                value["interaction"]["state"]["phase"] = "listening"
            return value
        self.assertEqual(self.run_prepare(startup=True, idle_probe=probe)["status"], "busy")
        self.assertEqual(self.factory_calls, [])

    def test_native_late_or_invalid_reply_never_becomes_ready(self):
        self.native_seconds = NATIVE_TIMEOUT_SECONDS + .1
        self.assertEqual(self.run_prepare()["reason"], "native_deadline_exceeded")
        self.setUp()
        for reply in (None, "", " ", "x" * 65, "你好", "好的", "好，我在", "PRIVATE_MODEL_BODY"):
            with self.subTest(reply_type=type(reply).__name__):
                self.reply = reply
                self.assertFalse(self.run_prepare()["ready"])

    def test_only_neutral_ack_with_optional_sentence_punctuation_is_ready(self):
        for reply in ("好", " 好。\n", "好!", "好！", "好.", "好？", "好?"):
            with self.subTest(reply=reply):
                self.reply = reply
                self.assertTrue(self.run_prepare()["ready"])

    def test_completed_body_does_not_mark_ready_when_lane_ack_is_unconfirmed(self):
        def probe(timeout):
            value = self.probe(timeout)
            if self.chat_calls:
                value["lane"]["state"] = "quarantined"
            return value
        result = self.run_prepare(idle_probe=probe)
        self.assertEqual(result["status"], "busy")
        self.assertEqual(result["stage"], "postflight")
        self.assertFalse(result["ready"])
        self.assertEqual(len(self.chat_calls), 1)

    def test_reply_and_status_payload_are_not_published(self):
        self.reply = "PRIVATE_MODEL_BODY"
        self.state["private"] = {"profile": "PRIVATE_PROFILE"}
        result = self.run_prepare()
        self.assertFalse(result["ready"])
        self.assertNotIn("PRIVATE", json.dumps(result))
        self.assertNotIn(WARMUP_TEXT, json.dumps(result))

    def test_invalid_health_bounds_rejected_before_probes(self):
        for value in (True, 0, -1, 10.1, float("nan"), float("inf"), "10"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.run_prepare(health_wait_seconds=value)


class FakeSocket:
    def __init__(self, parts):
        self.parts = list(parts)
        self.sent, self.connected, self.timeouts = [], [], []
    def __enter__(self):
        return self
    def __exit__(self, *_args):
        return False
    def settimeout(self, value):
        self.timeouts.append(value)
    def connect(self, value):
        self.connected.append(value)
    def sendall(self, value):
        self.sent.append(value)
    def recv(self, _maximum):
        return self.parts.pop(0) if self.parts else b""


class PreparationProbeTests(unittest.TestCase):
    def test_http_status_is_fixed_loopback_bounded_and_content_length_aware(self):
        body = json.dumps(idle_status()["interaction"]["state"]).encode()
        fake = FakeSocket([b"HTTP/1.0 200 OK\r\nContent-Length: " + str(len(body)).encode()
                           + b"\r\n\r\n" + body[:2], body[2:]])
        with patch("gateway.llm.preparation.socket.create_connection", return_value=fake) as connect:
            status, result = _http_get(8765, "/api/interaction/state", .5)
        self.assertEqual((status, result), (200, body))
        self.assertEqual(connect.call_args.args[0], ("127.0.0.1", 8765))
        self.assertTrue(fake.sent[0].startswith(b"GET /api/interaction/state "))
        self.assertTrue(all(0 < timeout <= .5 for timeout in fake.timeouts))

    def test_unapproved_http_endpoint_is_denied_before_network(self):
        with patch("socket.create_connection") as connect:
            with self.assertRaises(ValueError):
                _http_get(9001, "/v1/chat/completions", .5)
            connect.assert_not_called()

    def test_http_bad_framing_oversize_or_incomplete_body_is_rejected(self):
        for wire in (b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nx",
                     b"HTTP/1.1 200 OK\r\nContent-Length: 999999\r\n\r\n",
                     b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
                     b"HTTP/1.1 200 OK\r\n" + b"x" * MAX_STATUS_BYTES):
            with self.subTest(prefix=wire[:35]):
                with patch("socket.create_connection", return_value=FakeSocket([wire])):
                    with self.assertRaises((ValueError, KeyError)):
                        _http_get(8765, "/api/interaction/state", .5)

    def test_interaction_only_refused_or_404_maps_to_not_started(self):
        for side_effect, returned, reason in ((ConnectionRefusedError(), None, "connection_refused"),
                                             (None, (404, b""), "not_found")):
            with patch("gateway.llm.preparation._http_get", side_effect=side_effect, return_value=returned):
                self.assertEqual(_interaction_probe(.5), {"availability": "not_started", "reason": reason})
        for error in (TimeoutError(), ValueError("bad JSON")):
            with patch("gateway.llm.preparation._http_get", side_effect=error):
                with self.assertRaises(type(error)):
                    _interaction_probe(.5)
        with patch("gateway.llm.preparation._http_get", return_value=(200, b"bad JSON")):
            with self.assertRaises(ValueError):
                _interaction_probe(.5)

    def test_unix_status_accepts_only_canonical_read_operations(self):
        payload = json.dumps({"ok": True, "result": {"state": "idle"}}).encode() + b"\n"
        fake = FakeSocket([payload])
        with patch("gateway.llm.preparation.os.getuid", return_value=1000), \
                patch("gateway.llm.preparation.socket.socket", return_value=fake):
            self.assertEqual(_unix_status("/run/user/1000/vq-control.sock", "llm.status", .5), {"state": "idle"})
        self.assertEqual(fake.connected, ["/run/user/1000/vq-control.sock"])
        self.assertEqual(json.loads(fake.sent[0]), {"operation": "llm.status"})
        with patch("socket.socket") as connect:
            for path, operation in (("/private/other.sock", "llm.status"),
                                    ("/run/user/1000/vq-control.sock", "llm.finish")):
                with self.assertRaises(ValueError):
                    _unix_status(path, operation, .5)
            connect.assert_not_called()

    def test_cli_publishes_only_safe_summary_and_maps_exit_status(self):
        for state, code in (("ready", 0), ("busy", 2), ("failure", 1)):
            summary = {"status": state, "ready": state == "ready"}
            with patch("gateway.llm.preparation.prepare_once", return_value=summary) as prepare, \
                    patch("builtins.print") as printed:
                self.assertEqual(main(["--once", "--startup"]), code)
            self.assertTrue(prepare.call_args.kwargs["startup"])
            self.assertEqual(json.loads(printed.call_args.args[0]), summary)


if __name__ == "__main__":
    unittest.main()
