"""最小 Capability Registry — Capability Contract v0.1.

维护 name -> (metadata, impl) 的映射, 提供注册 / 列举 / 描述。
"""

from .errors import CapabilityNotFoundError


class Registry:
    def __init__(self):
        self._caps = {}  # name -> {"metadata": dict, "impl": callable}

    def register(self, name, metadata, impl):
        if not callable(impl):
            raise ValueError(f"impl 必须可调用: {name}")
        self._caps[name] = {"metadata": dict(metadata), "impl": impl}
        return self

    def get(self, name):
        entry = self._caps.get(name)
        if entry is None:
            raise CapabilityNotFoundError(
                f"capability 不存在: {name}", capability=name)
        return entry

    def list_capabilities(self):
        return sorted(self._caps.keys())

    def describe(self, name):
        return self.get(name)["metadata"]
