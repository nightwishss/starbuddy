"""vq-platform 最小 Capability Runtime (v0.1).

当前只含: Registry / Schema Validation / Dispatcher / 统一 Error Model。
用于验证 Command 型 Capability 的接口设计, 非最终架构。
"""

from .errors import (
    CapabilityError,
    CapabilityNotFoundError,
    SchemaValidationError,
    PermissionDeniedError,
    TimeoutError,
    ImplementationError,
    HardwareError,
    ResourceNotFoundError,
)
from .registry import Registry
from .dispatcher import execute_capability
from .router_rpc import (
    RouterRpcClient,
    RouterRpcError,
    RouterRpcProtocolError,
    RouterRpcRemoteError,
)

__all__ = [
    "Registry",
    "execute_capability",
    "CapabilityError",
    "CapabilityNotFoundError",
    "SchemaValidationError",
    "PermissionDeniedError",
    "TimeoutError",
    "ImplementationError",
    "HardwareError",
    "ResourceNotFoundError",
    "RouterRpcClient",
    "RouterRpcError",
    "RouterRpcProtocolError",
    "RouterRpcRemoteError",
]
