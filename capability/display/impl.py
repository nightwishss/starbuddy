"""Display Capability 实现 — Resource/Stateful 模型.

资源生命周期 (模块级 display_id -> backend 实例映射):
  display.open()   -> 创建 FramebufferBackend, 返回 display_id
  display.show_*() -> 按 display_id 找到 backend 操作
  display.close()  -> 销毁 backend 并从映射移除

v0.1 资源注册表放在 capability 内部 (模块级 dict), 不侵入共享 Runtime:
  - LED 的 command/query 无跨调用持久状态, 不需要资源表;
  - 只有 display (及未来 audio stream) 需要, 属 capability 内聚状态;
  - 等 audio 也验证后, 再决定是否上提到 Runtime (见 contract v0.2 draft)。
"""
import os
import uuid

from runtime.errors import ResourceNotFoundError
from .backend import FramebufferBackend, X11Backend

_DISPLAYS = {}  # display_id -> FramebufferBackend


def _get(display_id):
    b = _DISPLAYS.get(display_id)
    if b is None:
        raise ResourceNotFoundError(
            f"display 不存在或已关闭: {display_id}",
            capability="display", details={"display_id": display_id})
    return b


def list_displays(args=None):
    backends = ["framebuffer"]
    if X11Backend.is_available():
        backends.append("x11")
    return {
        "backends": backends,
        "active": sorted(_DISPLAYS.keys()),
        "count": len(_DISPLAYS),
    }


def get_info(args=None):
    return {
        "backend": "framebuffer",
        "backends": ["framebuffer"] + (["x11"] if X11Backend.is_available() else []),
        "type": "software",
        "physical": False,
        "max_width": 4096,
        "max_height": 4096,
        "formats": ["png"],
    }


def open_display(args):
    width = args.get("width", 1280)
    height = args.get("height", 720)
    backend_name = args.get("backend", "framebuffer")
    display_id = uuid.uuid4().hex[:8]
    if backend_name == "x11":
        backend = X11Backend(
            width, height,
            display=args.get("display") or os.environ.get("DISPLAY"),
            title=args.get("title", "VQ Display"),
        )
    else:
        backend = FramebufferBackend(width, height)
    _DISPLAYS[display_id] = backend
    return {
        "display_id": display_id, "width": width,
        "height": height, "backend": backend.kind,
    }


def show_text(args):
    b = _get(args["display_id"])
    b.show_text(args["text"], position=args.get("position"))
    return {"display_id": args["display_id"], "text": args["text"], "rendered": True}


def show_image(args):
    b = _get(args["display_id"])
    b.show_image(args["path"])
    return {"display_id": args["display_id"], "path": args["path"], "rendered": True}


def clear_display(args):
    b = _get(args["display_id"])
    b.clear(args.get("color", "black"))
    return {"display_id": args["display_id"], "cleared": True}


def capture(args):
    b = _get(args["display_id"])
    b.capture(args["path"])
    return {"display_id": args["display_id"], "path": args["path"]}


def close_display(args):
    b = _get(args["display_id"])
    b.close()
    del _DISPLAYS[args["display_id"]]
    return {"display_id": args["display_id"], "closed": True}


# action 名 -> 实现函数 (由 capability/display/__init__.py 注册进 Registry)
ACTIONS = {
    "display.list": list_displays,
    "display.get_info": get_info,
    "display.open": open_display,
    "display.show_text": show_text,
    "display.show_image": show_image,
    "display.clear": clear_display,
    "display.capture": capture,
    "display.close": close_display,
}
