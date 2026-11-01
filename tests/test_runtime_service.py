#!/usr/bin/env python3
"""Runtime Engine和Unix Socket服务的无硬件测试。"""

import asyncio
import json
import os
import stat
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime.client import RuntimeClient  # noqa: E402
from runtime.policy import AccessPolicy  # noqa: E402
from runtime.registry import Registry  # noqa: E402
from runtime.service import RuntimeEngine, RuntimeServer  # noqa: E402


def build_test_registry():
    registry = Registry()
    state = {"open": set(), "closed": []}
    registry.register("test.echo", {
        "name": "test.echo", "type": "query",
        "permissions": ["test.read"], "timeout_ms": 1000,
        "input": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"], "additionalProperties": False,
        },
    }, lambda args: {"value": args["value"]})

    def slow_open(args):
        time.sleep(0.02)
        state["open"].add("slow-one")
        return {"display_id": "slow-one"}

    def close_display(args):
        state["open"].discard(args["display_id"])
        state["closed"].append(args["display_id"])
        return {"closed": True}

    registry.register("display.open", {
        "name": "display.open", "type": "stateful",
        "permissions": ["test.write"], "timeout_ms": 1,
    }, slow_open)
    registry.register("display.close", {
        "name": "display.close", "type": "stateful",
        "permissions": ["test.write"], "timeout_ms": 1000,
    }, close_display)
    registry.register("test.large", {
        "name": "test.large", "type": "query",
        "permissions": ["test.read"], "timeout_ms": 1000,
    }, lambda args: {"data": "x" * (1024 * 1024)})
    return registry, state


async def main_async():
    checks = []

    def check(cond, label):
        checks.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    uid = os.getuid()
    registry, state = build_test_registry()
    engine = RuntimeEngine(
        registry, AccessPolicy(uid, {"test.read", "test.write"}))

    ok = await engine.handle({
        "id": "one", "capability": "test.echo",
        "args": {"value": "hello"}}, uid)
    check(ok["ok"] and ok["result"] == {"value": "hello"}, "Engine成功调用")
    check(ok["meta"]["elapsed_ms"] >= 0, "响应包含耗时")

    bad_schema = await engine.handle({
        "id": "two", "capability": "test.echo", "args": {}}, uid)
    check(not bad_schema["ok"] and
          bad_schema["error"]["code"] == "SCHEMA_VALIDATION_ERROR",
          "Capability Schema错误结构化返回")

    bad_peer = await engine.handle({
        "id": "three", "capability": "test.echo",
        "args": {"value": "x"}}, uid + 1)
    check(not bad_peer["ok"] and bad_peer["error"]["code"] == "PERMISSION_DENIED",
          "拒绝其他uid")

    unknown_field = await engine.handle({
        "id": "four", "capability": "test.echo",
        "args": {"value": "x"}, "extra": True}, uid)
    check(not unknown_field["ok"] and
          unknown_field["error"]["code"] == "SCHEMA_VALIDATION_ERROR",
          "拒绝未知协议字段")

    timed_out = await engine.handle({
        "id": "five", "capability": "display.open", "args": {}}, uid)
    check(not timed_out["ok"] and timed_out["error"]["code"] == "TIMEOUT",
          "超时调用返回TIMEOUT")
    check(not state["open"] and state["closed"] == ["slow-one"],
          "超时stateful open主动关闭资源")

    restricted = RuntimeEngine(registry, AccessPolicy(uid, {"test.read"}))
    denied_capability = await restricted.handle({
        "id": "six", "capability": "display.open", "args": {}}, uid)
    check(not denied_capability["ok"] and
          denied_capability["error"]["code"] == "PERMISSION_DENIED",
          "强制执行Capability权限allowlist")

    with tempfile.TemporaryDirectory() as tmp:
        socket_path = Path(tmp) / "runtime.sock"
        server = RuntimeServer(socket_path, engine)
        await server.start()
        serve_task = asyncio.create_task(server.server.serve_forever())
        try:
            mode = stat.S_IMODE(socket_path.stat().st_mode)
            check(mode == 0o600, "Unix Socket权限为0600")

            response = await asyncio.to_thread(
                RuntimeClient(socket_path).call,
                "test.echo", {"value": "socket"})
            check(response["ok"] and response["result"]["value"] == "socket",
                  "CLI客户端端到端调用")

            large = await asyncio.to_thread(
                RuntimeClient(socket_path).call, "test.large")
            check(not large["ok"] and
                  large["error"]["code"] == "IMPLEMENTATION_ERROR",
                  "超限响应替换为结构化错误")

            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            writer.write(b"not-json\n")
            await writer.drain()
            invalid = json.loads(await reader.readline())
            check(not invalid["ok"] and
                  invalid["error"]["code"] == "SCHEMA_VALIDATION_ERROR",
                  "非法JSON结构化返回")
            writer.close()
            await writer.wait_closed()
        finally:
            serve_task.cancel()
            try:
                await serve_task
            except asyncio.CancelledError:
                pass
            await server.close()
        check(not socket_path.exists(), "服务关闭后删除Socket")

    passed = sum(checks)
    print(f"\n=== 结果: {passed}/{len(checks)} PASS ===")
    raise SystemExit(0 if passed == len(checks) else 1)


if __name__ == "__main__":
    asyncio.run(main_async())
