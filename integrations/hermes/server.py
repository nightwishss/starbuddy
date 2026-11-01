"""Restricted MCP adapter from Hermes to the VQ Runtime Unix socket.

Every static Runtime capability is exposed as a first-class MCP tool with the
same JSON input schema. The adapter does not implement capabilities, execute a
shell, or access devices directly: all calls still cross the reviewed Runtime
policy boundary. Two legacy generic tools remain for compatibility. One fixed
situation-metadata hook does not enable a Hermes loop or sensor/profile access.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from typing import Any
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from runtime.catalog import build_registry
from runtime.client import RuntimeClient
from interaction_context.situation import TASKS


LEGACY_LIST_TOOL = "runtime_list_capabilities"
LEGACY_CALL_TOOL = "runtime_call_capability"
SITUATION_READ_TOOL = "starbuddy_situation_read"
_TOOL_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")
_SITUATION_URL = "http://127.0.0.1:8765/api/agent/state"
_SITUATION_MAX_BYTES = 65536
_PUBLIC_TASKS = {row["id"]: {key: row[key] for key in ("id", "title", "prompt")} for row in TASKS}


class _NoSituationRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A fixed loopback read must never become a request to another address.
        return None


def _public_situation(payload: Any) -> dict[str, Any]:
    """Only fixed enums, bounded numbers and trusted public task metadata.

    Never project response, transcript, input, actor label, identity, media,
    profile, token, history or arbitrary server error/decision text.
    """
    if (type(payload) is not dict or payload.get("ok") is not True
            or type(payload.get("state")) is not dict):
        raise ValueError("situation_response_invalid")
    source = payload["state"]
    state: dict[str, Any] = {}

    def enums(src, dst, key, allowed):
        value = src.get(key)
        if type(value) is str and value in allowed:
            dst[key] = value

    def booleans(src, dst, keys):
        for key in keys:
            if type(src.get(key)) is bool:
                dst[key] = src[key]

    def number(src, dst, key, maximum, *, integer=False):
        value = src.get(key)
        types_allowed = (int,) if integer else (int, float)
        if type(value) in types_allowed and 0 <= value <= maximum and math.isfinite(value):
            dst[key] = value

    number(source, state, "revision", 2 ** 53 - 1, integer=True)
    enums(source, state, "state", {"idle", "active", "completed"})
    enums(source, state, "phase", {"idle", "thinking", "listening", "speaking", "acting",
                                     "error", "quiet", "unavailable", "active", "completed"})
    enums(source, state, "reason", {"no_task", "clock_invalid", "clock_reset", "task_expired",
                                      "turn_expired", "task_selected", "task_cleared", "cancelled",
                                      "offers_changed", "turn_started", "answer_checked",
                                      "user_declined", "decision_recorded"})
    booleans(source, state, ("turn_pending", "offers_enabled", "context_is_live"))
    number(source, state, "expires_in_s", 900)
    raw_task = source.get("task")
    task_id = raw_task.get("id") if type(raw_task) is dict else None
    state["task"] = dict(_PUBLIC_TASKS[task_id]) if type(task_id) is str and task_id in _PUBLIC_TASKS else None
    for name in ("attempts", "offers", "decision", "privacy", "timing_ms"):
        if type(source.get(name)) is not dict:
            continue
        raw, value = source[name], {}
        if name == "attempts":
            number(raw, value, "correct", 999, integer=True)
            number(raw, value, "incorrect", 999, integer=True)
            if type(raw.get("recent")) is list:
                value["recent"] = [item for item in raw["recent"][-8:] if type(item) is bool]
        elif name == "offers":
            booleans(raw, value, ("enabled", "declined", "offered", "eligible"))
            enums(raw, value, "reason", {"no_task", "disabled", "user_declined", "task_complete",
                                         "already_offered", "insufficient_attempts", "eligible"})
        elif name == "decision":
            enums(raw, value, "action", {"respond", "answer", "clarify", "offer_help", "wait",
                                         "acknowledge", "complete", "cancel"})
            enums(raw, value, "intent", {"task_help", "task_answer", "task_selection", "decline_help",
                                         "help_task", "answer_task", "clarify", "inspect_scene", "chat",
                                         "cancel", "social", "unknown"})
            enums(raw, value, "reason", {"explicit_request", "task_selected", "answer_checked", "uncertain",
                                         "missing_task", "offer_eligible", "user_declined", "cancelled",
                                         "task_complete", "no_action", "model_decision"})
            booleans(raw, value, ("needs_clarification",))
            # Display uses a title; restore the trusted ID instead of forwarding text.
            if state["task"] is not None:
                value["target"] = task_id
        elif name == "privacy":
            booleans(raw, value, ("persistent", "profile_read", "automatic_capture"))
            enums(raw, value, "speaker_binding", {"unconfirmed"})
        elif name == "timing_ms":
            for key in ("context", "reasoning", "total_to_output_queue"):
                number(raw, value, key, 120000)
            number(raw, value, "model_calls", 1, integer=True)
        state[name] = value
    raw_backend = payload.get("backend")
    backend: dict[str, Any] = {}
    if type(raw_backend) is dict:
        enums(raw_backend, backend, "backend", {"local-bounded", "hermes-isolated", "unavailable", "injected-test"})
        enums(raw_backend, backend, "framework", {"hermes"})
        enums(raw_backend, backend, "framework_blocker", {"context_window_unverified_and_private_bootstrap"})
        enums(raw_backend, backend, "deadline_enforcement", {"discard_late_results"})
        booleans(raw_backend, backend, ("available", "hermes_available", "busy", "framework_enabled",
                                      "automatic_tools", "memory_access", "cloud_fallback"))
        number(raw_backend, backend, "max_model_calls", 1, integer=True)
    return {"ok": True, "state": state, "backend": backend,
            "scope": "public_situation_metadata_only", "raw_inputs_included": False}


def read_situation() -> dict[str, Any]:
    """One bounded GET to a fixed local endpoint, without redirects or proxies."""
    try:
        opener = build_opener(ProxyHandler({}), _NoSituationRedirect())
        with opener.open(_SITUATION_URL, timeout=2.0) as response:
            if response.status != 200:
                raise ValueError("situation_response_invalid")
            body = response.read(_SITUATION_MAX_BYTES + 1)
            if len(body) > _SITUATION_MAX_BYTES:
                raise ValueError("situation_response_too_large")
        return _public_situation(json.loads(body))
    except Exception as exc:
        # Never relay response bytes, traces or server-supplied text.
        raise RuntimeError("situation_read_unavailable") from exc


def configured_capability_allowlist() -> frozenset[str] | None:
    raw = os.environ.get("VQ_MCP_ALLOW_CAPABILITIES", "").strip()
    if not raw:
        return None
    return frozenset(item.strip() for item in raw.split(",") if item.strip())


CAPABILITY_ALLOWLIST = configured_capability_allowlist()


def capability_tool_name(capability: str) -> str:
    """Convert a dotted Runtime name to a portable MCP tool name."""
    return _TOOL_NAME_RE.sub("_", capability)


def capability_catalog(allowlist: frozenset[str] | None = CAPABILITY_ALLOWLIST) -> list[dict[str, Any]]:
    """Return model-facing metadata without implementation import paths."""
    registry = build_registry()
    result = []
    for name in registry.list_capabilities():
        if allowlist is not None and name not in allowlist:
            continue
        metadata = registry.describe(name)
        if metadata.get("expose_to_agent") is False:
            continue
        result.append({
            "name": name,
            "tool_name": capability_tool_name(name),
            "description": metadata.get("description", ""),
            "type": metadata.get("type", ""),
            "input_schema": metadata.get("input", {"type": "object"}),
            "timeout_ms": metadata.get("timeout_ms"),
        })
    return result


def build_tool_catalog(allowlist: frozenset[str] | None = CAPABILITY_ALLOWLIST, *,
                       include_situation: bool = False) -> tuple[list[types.Tool], dict[str, str]]:
    """Build MCP definitions and Runtime routing, preserving the legacy core.

    The live server includes the separate public-situation hook; callers asking
    for a Runtime-only catalogue retain their original contract.
    """
    tools = [
        types.Tool(
            name=LEGACY_LIST_TOOL,
            description="列出VENTUNO Q板卡的全部Runtime能力、说明与参数Schema。",
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
            annotations=types.ToolAnnotations(
                readOnlyHint=True, destructiveHint=False,
                idempotentHint=True, openWorldHint=False),
        ),
        types.Tool(
            name=LEGACY_CALL_TOOL,
            description="兼容入口：按Runtime capability名称调用板卡能力。优先使用对应的独立工具。",
            inputSchema={
                "type": "object",
                "properties": {
                    "capability": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["capability"],
                "additionalProperties": False,
            },
            annotations=types.ToolAnnotations(openWorldHint=False),
        ),
    ]
    routes: dict[str, str] = {}
    for item in capability_catalog(allowlist):
        tool_name = item["tool_name"]
        if tool_name in routes or tool_name in {LEGACY_LIST_TOOL, LEGACY_CALL_TOOL, SITUATION_READ_TOOL}:
            raise RuntimeError(f"MCP工具名冲突: {tool_name}")
        routes[tool_name] = item["name"]
        read_only = item["type"] == "query"
        tools.append(types.Tool(
            name=tool_name,
            description=(
                f"VENTUNO Q板卡能力：{item['description']} "
                f"真实执行Runtime capability `{item['name']}`；不要只描述计划。"
            ),
            inputSchema=item["input_schema"],
            annotations=types.ToolAnnotations(
                readOnlyHint=read_only,
                destructiveHint=False if read_only else None,
                idempotentHint=True if read_only else None,
                openWorldHint=False,
            ),
        ))
    if include_situation:
        tools.append(types.Tool(
            name=SITUATION_READ_TOOL,
            description=("只读芽芽当前选中的学习任务、互动状态、结构化决策及后端就绪元数据。"
                         "这是Hermes框架的只读接入钩子，不表示Hermes推理循环已启用；"
                         "不读取个人档案、原始输入或媒体，不启动感知，不执行动作。"),
            inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
            annotations=types.ToolAnnotations(readOnlyHint=True, destructiveHint=False,
                                              idempotentHint=True, openWorldHint=False),
        ))
    return tools, routes


def call_runtime(capability: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Forward one call and turn Runtime errors into MCP tool errors."""
    socket_path = os.environ.get("VQ_RUNTIME_SOCKET")
    response = RuntimeClient(socket_path=socket_path).call(capability, arguments or {})
    if not response.get("ok"):
        error = response.get("error") or {}
        code = error.get("code", "runtime_error")
        message = error.get("message", "VQ Runtime call failed")
        raise RuntimeError(f"{code}: {message}")
    return {"result": response.get("result"), "meta": response.get("meta", {})}


def ensure_capability_allowed(capability: str) -> None:
    if CAPABILITY_ALLOWLIST is not None and capability not in CAPABILITY_ALLOWLIST:
        raise RuntimeError(f"本地Agent不允许调用Runtime capability: {capability}")
    try:
        metadata = build_registry().describe(capability)
    except Exception as exc:
        raise RuntimeError(f"未知Runtime capability: {capability}") from exc
    if metadata.get("expose_to_agent") is False:
        raise RuntimeError(f"Agent无权调用内部Runtime capability: {capability}")


TOOLS, TOOL_ROUTES = build_tool_catalog(include_situation=True)
server = Server(
    "vq-runtime",
    version="0.2.0",
    instructions=(
        "Use the explicit VENTUNO Q capability tools whenever an answer depends "
        "on current board state or must change the board. Never claim an action "
        "succeeded without a successful tool result. The Runtime authorizes, "
        "validates, owns resources, times out, and audits capability operations. "
        "The separate starbuddy_situation_read hook reads only public local "
        "interaction metadata; it does not enable Hermes reasoning, capture "
        "sensors, authorize personal profiles, or execute actions."
    ),
)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return TOOLS


@server.call_tool(validate_input=True)
async def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == SITUATION_READ_TOOL:
        if type(arguments) is not dict or arguments:
            raise ValueError("situation_read_accepts_no_arguments")
        return await asyncio.to_thread(read_situation)
    if name == LEGACY_LIST_TOOL:
        return {"capabilities": capability_catalog(CAPABILITY_ALLOWLIST)}
    if name == LEGACY_CALL_TOOL:
        capability = arguments["capability"]
        runtime_args = arguments.get("arguments") or {}
    else:
        try:
            capability = TOOL_ROUTES[name]
        except KeyError as exc:
            raise RuntimeError(f"未知VQ MCP工具: {name}") from exc
        runtime_args = arguments
    ensure_capability_allowed(capability)
    return await asyncio.to_thread(call_runtime, capability, runtime_args)


async def async_main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
