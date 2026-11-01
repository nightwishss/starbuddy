"""Service entrypoint for task events to StarBuddy embodied output."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .consumer import TaskEventConsumer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=os.environ.get(
        "VQ_TASK_INTERACTION_STATE",
        str(Path.home() / ".local/state/vq-platform/task-interaction.json"),
    ))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    from event_service import EventClient
    from gateway.display import DisplayClient
    from output_service import OutputClient
    consumer = TaskEventConsumer(
        EventClient(timeout=25), DisplayClient(timeout=1.0), OutputClient(timeout=35),
        Path(args.state),
    )
    try:
        consumer.run_forever()
    except KeyboardInterrupt:
        consumer.stop()


if __name__ == "__main__":
    main()
