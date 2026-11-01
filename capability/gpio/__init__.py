"""GPIO capability 包: 加载 metadata.json 并注册进 Registry。"""

import json
from pathlib import Path

from . import impl

_METADATA = json.loads(
    (Path(__file__).parent / "metadata.json").read_text(encoding="utf-8"))


def register(registry):
    """把 gpio 的全部 capability 注册进给定 Registry。"""
    for meta in _METADATA["capabilities"]:
        name = meta["name"]
        registry.register(name, meta, impl.ACTIONS[name])
    return registry
