#!/usr/bin/env python3
"""Protocol-level smoke test for Hermes -> MCP -> VQ Runtime."""

import asyncio
import json
import os
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]


async def run():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env.setdefault("VQ_RUNTIME_SOCKET", "/run/user/1000/vq-runtime.sock")
    params = StdioServerParameters(
        command=os.environ.get("VQ_HERMES_PYTHON", os.sys.executable),
        args=[str(PROJECT_ROOT / "integrations/hermes/server.py")],
        env=env,
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            expected = {
                "runtime_list_capabilities", "runtime_call_capability",
                "system_info", "system_status", "led_set_rgb",
            }
            missing = expected - names
            if missing:
                raise RuntimeError(f"missing MCP tools: {sorted(missing)}")
            result = await session.call_tool(
                "system_info",
                {},
            )
            if result.isError:
                raise RuntimeError(f"system.info failed: {result.content}")
            print(json.dumps({
                "tools": sorted(names),
                "system_info": result.structuredContent,
            }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
