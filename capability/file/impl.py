"""File Capability 实现 — Capability Contract v0.1.

受限路径下的文件读写 (纯软件, stdlib)。安全约束:
  - 所有路径 resolve 后必须落在 ALLOWED_ROOT 内 (默认 ~), 阻止路径穿越 (..)。
  - 只暴露文本读写 + 列目录 + 存在性检查 (二进制/权限变更不在 v0.1)。
"""
import os
from pathlib import Path

from runtime.errors import HardwareError, SchemaValidationError

ALLOWED_ROOT = Path.home().resolve()


def _resolve(path):
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    r = p.resolve()
    if r != ALLOWED_ROOT and not str(r).startswith(str(ALLOWED_ROOT) + os.sep):
        raise SchemaValidationError(
            f"路径越界: {path} 超出允许根 {ALLOWED_ROOT}",
            capability="file", details={"path": path})
    return r


def read(args):
    r = _resolve(args["path"])
    try:
        content = r.read_text(encoding="utf-8")
    except OSError as e:
        raise HardwareError(
            f"读文件失败 {r}: {e!r}", capability="file", details={"path": str(r)}) from e
    return {"path": str(r), "content": content, "bytes": len(content.encode("utf-8"))}


def write(args):
    r = _resolve(args["path"])
    try:
        r.parent.mkdir(parents=True, exist_ok=True)
        r.write_text(args["content"], encoding="utf-8")
    except OSError as e:
        raise HardwareError(
            f"写文件失败 {r}: {e!r}", capability="file", details={"path": str(r)}) from e
    return {"path": str(r), "bytes": len(args["content"].encode("utf-8"))}


def list_dir(args):
    r = _resolve(args["path"])
    try:
        entries = [
            {"name": p.name, "type": "dir" if p.is_dir() else "file",
             "size": p.stat().st_size}
            for p in sorted(r.iterdir())
        ]
    except OSError as e:
        raise HardwareError(
            f"列目录失败 {r}: {e!r}", capability="file", details={"path": str(r)}) from e
    return {"path": str(r), "count": len(entries), "entries": entries}


def exists(args):
    r = _resolve(args["path"])
    return {"path": str(r), "exists": r.exists()}


# action 名 -> 实现函数 (由 capability/file/__init__.py 注册进 Registry)
ACTIONS = {
    "file.read": read,
    "file.write": write,
    "file.list": list_dir,
    "file.exists": exists,
}
