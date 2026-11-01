#!/usr/bin/env python3
"""VQ Runtime调试CLI。"""

import argparse
import json
from pathlib import Path

from runtime.client import RuntimeClient


def main():
    parser = argparse.ArgumentParser(prog="vqctl")
    parser.add_argument("capability")
    parser.add_argument("--args", default="{}", help="JSON object")
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--timeout-ms", type=int)
    ns = parser.parse_args()
    args = json.loads(ns.args)
    response = RuntimeClient(ns.socket).call(
        ns.capability, args=args, timeout_ms=ns.timeout_ms)
    print(json.dumps(response, ensure_ascii=False, indent=2))
    raise SystemExit(0 if response.get("ok") else 1)


if __name__ == "__main__":
    main()
