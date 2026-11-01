"""Command-line adapter for UI-free conversation session access."""

from __future__ import annotations

import argparse
import json

from .client import ConversationClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("open", "context", "reset"):
        child = sub.add_parser(command)
        child.add_argument("session_id")
    start = sub.add_parser("start")
    start.add_argument("session_id")
    start.add_argument("text")
    start.add_argument("--input-type", default="text")
    complete = sub.add_parser("complete")
    complete.add_argument("session_id")
    complete.add_argument("turn_id")
    complete.add_argument("text")
    complete.add_argument("--backend", default="manual")
    remember = sub.add_parser("remember")
    remember.add_argument("session_id")
    remember.add_argument("key")
    remember.add_argument("json_value")
    remember.add_argument("--source", default="cli")
    forget = sub.add_parser("forget")
    forget.add_argument("session_id")
    forget.add_argument("key")
    sub.add_parser("stats")
    args = parser.parse_args()
    client = ConversationClient(args.socket) if args.socket else ConversationClient()
    if args.command == "open":
        result = client.ensure(args.session_id)
    elif args.command == "context":
        result = client.context(args.session_id)
    elif args.command == "reset":
        result = client.reset(args.session_id)
    elif args.command == "start":
        result = client.start_turn(args.session_id, args.text, input_type=args.input_type)
    elif args.command == "complete":
        result = client.complete_turn(args.session_id, args.turn_id, args.text, backend=args.backend)
    elif args.command == "remember":
        result = client.remember(args.session_id, args.key, json.loads(args.json_value), source=args.source)
    elif args.command == "forget":
        result = client.forget(args.session_id, args.key)
    else:
        result = client.stats()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
