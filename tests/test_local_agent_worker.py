import json
import unittest
from unittest.mock import Mock

from gateway.agent.worker import AgentError, LocalAgent, parse_decision


class LocalAgentWorkerTests(unittest.TestCase):
    def test_parse_decision_accepts_fenced_json(self):
        self.assertEqual(parse_decision('```json\n{"action":"tool","name":"system.status","arguments":{}}\n```')["name"], "system.status")

    def test_parse_decision_rejects_free_text(self):
        with self.assertRaises(AgentError):
            parse_decision("I will check it")

    def test_agent_plans_executes_and_observes(self):
        model = Mock()
        model.timeout = 30.0
        model.complete.side_effect = [
            '{"action":"tool","name":"system.status","arguments":{}}',
            '{"action":"answer","content":"温度正常。"}',
        ]
        runtime = Mock()
        runtime.call.return_value = {"ok": True, "result": {"temperature": {"max_c": 36.0}}}
        agent = LocalAgent(model=model, runtime=runtime)
        result = agent.chat("检查温度")
        self.assertEqual(result["answer"], "温度正常。")
        self.assertEqual(result["steps"], 1)
        self.assertRegex(result["request_id"], r"^local-agent-[0-9a-f]{32}$")
        call = runtime.call.call_args
        self.assertEqual(call.args, ("system.status", {}))
        self.assertEqual(call.kwargs["timeout_ms"], 10_000)
        self.assertRegex(call.kwargs["request_id"], r"^local-agent-[0-9a-f]{32}-step-1$")
        self.assertIn('"max_c": 36.0', model.complete.call_args_list[1].args[0][1]["content"])

    def test_agent_denies_unlisted_tool(self):
        model = Mock()
        model.timeout = 30.0
        model.complete.return_value = json.dumps({"action": "tool", "name": "file.write", "arguments": {}})
        with self.assertRaisesRegex(AgentError, "denied"):
            LocalAgent(model=model, runtime=Mock()).chat("写文件")

    def test_matrix_status_is_available_as_read_only_tool(self):
        model = Mock()
        model.timeout = 30.0
        model.complete.side_effect = [
            '{"action":"tool","name":"matrix.status","arguments":{}}',
            '{"action":"answer","content":"矩阵已就绪。"}',
        ]
        runtime = Mock()
        runtime.call.return_value = {"ok": True, "result": {"ready": True, "rows": 8, "columns": 13}}
        result = LocalAgent(model=model, runtime=runtime).chat("矩阵是否正常")
        self.assertEqual(result["answer"], "矩阵已就绪。")
        self.assertEqual(runtime.call.call_args.args, ("matrix.status", {}))

    def test_parser_rejects_extra_fields(self):
        with self.assertRaisesRegex(AgentError, "unexpected"):
            parse_decision('{"action":"tool","name":"system.status","arguments":{},"shell":"id"}')

    def test_parser_rejects_truncated_and_multiple_objects(self):
        for value in ('{"action":"tool"', '{} {}', '```json\nnot-json\n```'):
            with self.subTest(value=value), self.assertRaises(AgentError):
                parse_decision(value)

    def test_duplicate_tool_call_is_stopped(self):
        model = Mock()
        model.timeout = 30.0
        model.complete.side_effect = [
            '{"action":"tool","name":"system.status","arguments":{}}',
            '{"action":"tool","name":"system.status","arguments":{}}',
        ]
        runtime = Mock()
        runtime.call.return_value = {"ok": True, "result": {"ok": True}}
        with self.assertRaisesRegex(AgentError, "duplicate"):
            LocalAgent(model=model, runtime=runtime).chat("重复查询")
        self.assertEqual(runtime.call.call_count, 1)


if __name__ == "__main__":
    unittest.main()
