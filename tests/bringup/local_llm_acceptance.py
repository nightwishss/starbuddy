#!/usr/bin/env python3
"""Non-destructive acceptance probe for the loopback Genie LLM service."""

import argparse
import json
import time
import urllib.error
import urllib.request


def request(base_url, path, *, payload=None, timeout=180):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read(4 * 1024 * 1024)
            return response.status, time.monotonic() - started, json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read(1024 * 1024)
        detail = raw.decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {detail}") from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:9001")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    checks = {}
    status, elapsed, health = request(args.base_url, "/v1/health", timeout=args.timeout)
    checks["health"] = {"status": status, "seconds": round(elapsed, 3), "body": health}

    status, elapsed, models = request(args.base_url, "/v1/models", timeout=args.timeout)
    data = models.get("data") or []
    if not data or not isinstance(data[0].get("id"), str):
        raise RuntimeError(f"no usable model returned: {models!r}")
    model = data[0]["id"]
    checks["models"] = {"status": status, "seconds": round(elapsed, 3), "selected": model}

    basic = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是简洁可靠的板端助手。"},
            {"role": "user", "content": "只用一句中文回答：你现在运行在哪里？"},
        ],
        "max_completion_tokens": 64,
        "temperature": 0.1,
        "stream": False,
    }
    status, elapsed, completion = request(
        args.base_url, "/v1/chat/completions", payload=basic, timeout=args.timeout
    )
    choices = completion.get("choices") or []
    content = ((choices[0].get("message") or {}).get("content") if choices else None)
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"empty basic completion: {completion!r}")
    checks["chinese_chat"] = {
        "status": status,
        "seconds": round(elapsed, 3),
        "content": content,
        "usage": completion.get("usage"),
    }

    tool_payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "必须调用 system_status 获取真实状态，不要直接回答。",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "system_status",
                    "description": "Read current CPU, memory, disk, and temperature.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "system_status"}},
        "max_completion_tokens": 128,
        "temperature": 0,
        "stream": False,
    }
    try:
        status, elapsed, completion = request(
            args.base_url, "/v1/chat/completions", payload=tool_payload, timeout=args.timeout
        )
        choices = completion.get("choices") or []
        message = choices[0].get("message") or {} if choices else {}
        tool_calls = message.get("tool_calls") or []
        valid_tool_call = False
        if len(tool_calls) == 1:
            function = tool_calls[0].get("function") or {}
            try:
                arguments = json.loads(function.get("arguments", ""))
            except (TypeError, json.JSONDecodeError):
                arguments = None
            valid_tool_call = function.get("name") == "system_status" and arguments == {}
        checks["native_tool_request"] = {
            "accepted": True,
            "valid_tool_call": valid_tool_call,
            "status": status,
            "seconds": round(elapsed, 3),
            "finish_reason": choices[0].get("finish_reason") if choices else None,
            "tool_calls": tool_calls,
            "content": message.get("content"),
        }
    except RuntimeError as exc:
        checks["native_tool_request"] = {"accepted": False, "valid_tool_call": False, "error": str(exc)}

    status, elapsed, metrics = request(args.base_url, "/v1/metrics", timeout=args.timeout)
    checks["metrics"] = {"status": status, "seconds": round(elapsed, 3), "body": metrics}

    print(json.dumps({"model": model, "checks": checks}, ensure_ascii=False, indent=2))
    if not checks["native_tool_request"]["valid_tool_call"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
