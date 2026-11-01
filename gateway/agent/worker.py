"""Persistent bounded Qwen Agent worker; loopback-only and Runtime-only."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import urllib.error
import urllib.request
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from runtime.catalog import build_registry
from runtime.client import RuntimeClient


LOG = logging.getLogger("vq.local_agent")
MAX_REQUEST_BYTES = 32 * 1024
MAX_MESSAGE_CHARS = 4000
MAX_STEPS = 3
MAX_TURN_SECONDS = 30.0
ALLOWLIST = frozenset({"system.status", "matrix.status"})


class AgentError(RuntimeError):
    pass


class QwenClient:
    def __init__(self, url="http://127.0.0.1:9001/v1/chat/completions", timeout=30.0):
        self.url = url
        self.timeout = timeout

    def complete(self, messages, max_tokens=192, timeout=None):
        payload = {"model": "qwen3_4b_instruct_2507", "messages": messages,
                   "max_completion_tokens": max_tokens, "temperature": 0.1, "stream": False}
        request = urllib.request.Request(self.url, data=json.dumps(payload, ensure_ascii=False).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                result = json.loads(response.read(2 * 1024 * 1024))
            return result["choices"][0]["message"]["content"].strip()
        except (OSError, urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise AgentError("local model unavailable") from exc


def parse_decision(text):
    value = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = value.find("{"), value.rfind("}")
    if start < 0:
        raise AgentError("model did not return a decision")
    candidate = value[start:] if end < start else value[start:end + 1]
    for suffix in ("", "}", "}}"):
        try:
            decision = json.loads(candidate + suffix)
            break
        except json.JSONDecodeError:
            decision = None
    if not isinstance(decision, dict) or decision.get("action") not in {"tool", "answer", "escalate"}:
        raise AgentError("invalid model decision")
    allowed_keys = {
        "tool": {"action", "name", "arguments"},
        "answer": {"action", "content"},
        "escalate": {"action", "reason"},
    }[decision["action"]]
    if set(decision) - allowed_keys:
        raise AgentError("unexpected decision fields")
    if decision["action"] == "tool" and (
        not isinstance(decision.get("name"), str)
        or not isinstance(decision.get("arguments"), dict)
    ):
        raise AgentError("invalid tool decision")
    if decision["action"] == "answer" and not isinstance(decision.get("content"), str):
        raise AgentError("invalid answer decision")
    if decision["action"] == "escalate" and not isinstance(decision.get("reason"), str):
        raise AgentError("invalid escalation decision")
    return decision


class LocalAgent:
    def __init__(self, model=None, runtime=None):
        self.model = model or QwenClient()
        self.runtime = runtime or RuntimeClient(timeout=10.0)
        registry = build_registry()
        self.tools = {name: registry.describe(name)["input"] for name in ALLOWLIST}
        self.lock = threading.Lock()

    def chat(self, message):
        if not self.lock.acquire(blocking=False):
            raise AgentError("local Agent busy")
        started = time.monotonic()
        request_id = "local-agent-" + uuid.uuid4().hex
        try:
            history = []
            called = set()
            for step in range(1, MAX_STEPS + 1):
                if time.monotonic() - started >= MAX_TURN_SECONDS:
                    raise AgentError("turn timeout")
                prompt = (
                    "你是VENTUNO Q板端的受限本地Agent。你负责理解、规划和选择工具，程序只校验并执行。"
                    "每轮只输出一个JSON对象，不加Markdown。格式三选一："
                    '{"action":"tool","name":"能力名","arguments":{}}；'
                    '{"action":"answer","content":"中文回答"}；'
                    '{"action":"escalate","reason":"原因"}。'
                    f"最多{MAX_STEPS}步。当前工具Schema：{json.dumps(self.tools, ensure_ascii=False)}。"
                    "涉及真实板卡状态必须调用工具，不得猜测；复杂或未开放任务选择escalate。"
                )
                context = "\n".join(history)
                remaining = max(0.1, MAX_TURN_SECONDS - (time.monotonic() - started))
                raw = self.model.complete(
                    [{"role": "system", "content": prompt},
                     {"role": "user", "content": f"用户请求：{message}\n已执行记录：{context}"}],
                    128, timeout=min(self.model.timeout if hasattr(self.model, "timeout") else remaining, remaining),
                )
                if time.monotonic() - started >= MAX_TURN_SECONDS:
                    raise AgentError("turn timeout")
                decision = parse_decision(raw)
                if decision["action"] == "answer":
                    content = decision.get("content")
                    if not isinstance(content, str) or not content.strip():
                        raise AgentError("empty answer")
                    return {"answer": content.strip(), "steps": step - 1,
                            "request_id": request_id,
                            "elapsed_ms": round((time.monotonic() - started) * 1000, 2)}
                if decision["action"] == "escalate":
                    reason = str(decision.get("reason") or "任务超出本地Agent能力范围")
                    return {"answer": f"需要切换到Hermes Agent：{reason}", "steps": step - 1,
                            "escalate": True, "request_id": request_id,
                            "elapsed_ms": round((time.monotonic() - started) * 1000, 2)}
                name, arguments = decision.get("name"), decision.get("arguments")
                if name not in ALLOWLIST or not isinstance(arguments, dict):
                    raise AgentError("tool denied")
                signature = (name, json.dumps(arguments, sort_keys=True, ensure_ascii=False))
                if signature in called:
                    raise AgentError("duplicate tool call")
                called.add(signature)
                response = self.runtime.call(
                    name, arguments, timeout_ms=10_000,
                    request_id=f"{request_id}-step-{step}",
                )
                if not response.get("ok"):
                    error = response.get("error") or {}
                    LOG.warning(
                        "Runtime rejected request_id=%s capability=%s argument_keys=%s code=%s path=%s",
                        request_id, name, sorted(arguments), error.get("code"),
                        (error.get("details") or {}).get("path"),
                    )
                    raise AgentError("Runtime tool failed")
                history.append(json.dumps({"tool": name, "result": response.get("result")}, ensure_ascii=False))
            raise AgentError("step limit reached")
        finally:
            self.lock.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        LOG.info("%s %s", self.client_address[0], fmt % args)

    def do_GET(self):
        if self.path != "/healthz":
            self.send_error(HTTPStatus.NOT_FOUND); return
        self._json(HTTPStatus.OK, {"ok": True, "capabilities": sorted(ALLOWLIST)})

    def do_POST(self):
        if self.path != "/v1/chat":
            self.send_error(HTTPStatus.NOT_FOUND); return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES: raise ValueError
            payload = json.loads(self.rfile.read(length))
            message = payload.get("message")
            if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_CHARS: raise ValueError
            result = self.server.agent.chat(message.strip())
            self._json(HTTPStatus.OK, result)
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid request"})
        except AgentError:
            LOG.exception("local Agent request failed")
            self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "local Agent unavailable"})

    def _json(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1"); parser.add_argument("--port", type=int, default=9003)
    args = parser.parse_args(); logging.basicConfig(level=logging.INFO)
    server = ThreadingHTTPServer((args.host, args.port), Handler); server.agent = LocalAgent()
    server.serve_forever()


if __name__ == "__main__": main()
