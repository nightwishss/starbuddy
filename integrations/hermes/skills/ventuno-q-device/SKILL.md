---
name: ventuno-q-device
description: "Use when the user asks Hermes to inspect or control the VENTUNO Q board, especially its onboard 8x13 blue LED matrix. Select the explicit VQ Runtime MCP tool, execute it, and report only the verified result."
version: 1.0.0
author: VQ Platform
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [ventuno-q, arduino, led-matrix, device-control, mcp]
    related_skills: []
---

# VENTUNO Q Device Control

## Overview

Use the first-class `vq_runtime` MCP tools to inspect or operate this physical board. Never use shell,
files, device nodes, the matrix relay socket, Arduino App internals, or RouterBridge directly. Runtime owns
authorization, schema validation, timeouts, auditing, and hardware errors.

## Matrix intent routing

- Check readiness or dimensions: call `matrix_status`.
- Show a built-in symbol: call `matrix_icon` with `name` and optional `brightness`.
- Blink a symbol: call `matrix_blink` with `name`, duration, interval, brightness, and final state.
- Turn the matrix off: call `matrix_clear`.
- Draw a custom frame only when all 104 row-major brightness values are available: call `matrix_draw`.

Built-in icon names are `check`, `cross`, `heart`, and `smile`. Brightness is 1 through 7 for icons.
Blink duration is 1 through 30 seconds, interval is 100 through 2000 milliseconds, and `end_state` is
`on` or `off`. Raw frames are exactly 8 rows by 13 columns (104 integers), each from 0 through 7.

## Execution rules

1. Prefer the explicit matrix tool over the compatibility-only `runtime_call_capability` tool.
2. Execute the tool; do not merely describe the intended action.
3. Do not claim physical success unless the tool result reports success. For first-time or safety-critical
   display changes, ask the user for visual confirmation after a successful result.
4. Do not emulate blinking with repeated Agent calls. Use the bounded `matrix_blink` capability.
5. If Runtime returns a schema, permission, timeout, or hardware error, report that error without inventing
   a device state.

## Skill and memory hygiene

- Treat this skill and the project runbooks as procedural source of truth; use `skill_view` instead of
  rediscovering the matrix workflow.
- Save only compact, stable user preferences, environment facts, and reusable lessons to `memory`.
- Never save passwords, API keys, tokens, pairing codes, certificates, private media, raw logs, or transient
  system readings to memory.
- Do not use memory as a task log or deployment record. Reviews and Git commits remain the authoritative
  record of completed work.

## Examples

- “显示一个心”：`matrix_icon({"name":"heart","brightness":5})`
- “让心闪烁10秒”：`matrix_blink({"name":"heart","brightness":5,"duration_seconds":10,"interval_ms":500,"end_state":"on"})`
- “把灯关掉”：`matrix_clear({})`
- “矩阵是否就绪”：`matrix_status({})`

## Common pitfalls

- The matrix is blue-only; brightness changes intensity, not color.
- Do not confuse this 8x13 matrix with the separate RGB status LED capabilities.
- Do not send 0 as icon brightness; use `matrix_clear` to turn it off.
- A successful MCP discovery is not a hardware write. Require a successful call result.

## Verification checklist

- [ ] The requested operation maps to an explicit `matrix_*` tool.
- [ ] Arguments satisfy the published schema.
- [ ] Runtime returned success.
- [ ] The response states what actually happened and does not expose internal transport details.
