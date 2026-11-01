"""构建VQ Runtime的标准Capability目录。"""

from runtime.registry import Registry


def build_registry():
    from capability import audio, camera, device, display, file, gpio, led, matrix, perception, speech, system, vision

    registry = Registry()
    for package in (system, device, file, gpio, led, matrix, display, audio, camera, speech, vision, perception):
        package.register(registry)
    return registry
