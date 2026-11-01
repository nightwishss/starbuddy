"""最小 Dispatcher — Capability Contract v0.1.

execute_capability(name, args) 的执行链:
  Registry 查找 -> Schema 校验输入 -> (权限检查, v0.1 仅声明) -> 调用 impl -> 错误统一映射 -> (输出校验)

v0.1 只验证 Command 型能力的接口; 权限强制 / 超时 / 生命周期 / 指标 属 Phase 3。
"""

from .errors import CapabilityError, ImplementationError
from .schema import validate


def execute_capability(registry, name, args=None):
    entry = registry.get(name)  # 不存在 -> CapabilityNotFoundError
    meta = entry["metadata"]
    args = dict(args or {})

    # 1. 输入 Schema 校验
    if meta.get("input"):
        validate(meta["input"], args, path=f"{name}.input")

    # 2. 权限检查 (v0.1 预留: 仅读取声明, 不实际拦截; 完整 RBAC 待 Phase 3)
    _declared_permissions(meta, name)

    # 3. 调用实现, 统一错误映射
    try:
        result = entry["impl"](args)
    except CapabilityError:
        raise
    except Exception as e:
        raise ImplementationError(
            f"{name} 实现内部错误: {e!r}", capability=name) from e

    # 4. 输出 Schema 校验 (可选)
    if meta.get("output") and result is not None:
        validate(meta["output"], result, path=f"{name}.output")

    return result


def _declared_permissions(meta, name):
    """v0.1: 只返回声明, 不做 caller 上下文的强制校验。"""
    return meta.get("permissions") or []
