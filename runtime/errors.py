"""统一错误模型 (Unified Error Model) — Capability Contract v0.1.

所有 capability 执行路径中的错误都映射为 CapabilityError 子类,
携带稳定的 code 字段, 供上层 (Hermes / Skill / Runtime) 统一处理,
而非暴露底层异常 (OSError 等)。
"""


class CapabilityError(Exception):
    """所有 capability 错误的基类。"""

    code = "CAPABILITY_ERROR"

    def __init__(self, message="", *, capability=None, details=None):
        self.message = message
        self.capability = capability
        self.details = details or {}
        super().__init__(message)

    def to_dict(self):
        return {
            "code": self.code,
            "message": self.message,
            "capability": self.capability,
            "details": self.details,
        }


class CapabilityNotFoundError(CapabilityError):
    """请求的 capability 未注册。"""

    code = "CAPABILITY_NOT_FOUND"


class SchemaValidationError(CapabilityError):
    """输入/输出不符合 schema。"""

    code = "SCHEMA_VALIDATION_ERROR"


class PermissionDeniedError(CapabilityError):
    """调用方缺少所需权限 (v0.1 仅声明, 未强制)。"""

    code = "PERMISSION_DENIED"


class TimeoutError(CapabilityError):
    """执行超时 (v0.1 仅声明, 未强制)。"""

    code = "TIMEOUT"


class ImplementationError(CapabilityError):
    """实现内部出错 (非硬件, 例如逻辑异常)。"""

    code = "IMPLEMENTATION_ERROR"


class HardwareError(CapabilityError):
    """硬件访问失败 (设备不存在 / 权限 / IO 错误)。"""

    code = "HARDWARE_ERROR"


class ResourceNotFoundError(CapabilityError):
    """资源句柄 (display_id / stream_id) 不存在或已关闭。

    Display/Audio 的 open→use→close 生命周期需要区分"这个资源句柄失效了"
    与"硬件坏了"两种错误，因此新增独立 code (最小增量, 不影响已有类)。
    """

    code = "RESOURCE_NOT_FOUND"
