"""Register board-local speech capabilities."""

import json
from pathlib import Path

from . import impl

_METADATA = json.loads((Path(__file__).parent / "metadata.json").read_text(encoding="utf-8"))


def register(registry):
    for meta in _METADATA["capabilities"]:
        registry.register(meta["name"], meta, impl.ACTIONS[meta["name"]])
    return registry
