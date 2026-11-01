"""LED Capability 实现 (Command 类型) — Capability Contract v0.1.

底层真实接口 (Phase 1 bring-up 已实测验证, 不重新猜测):
  /sys/class/leds/pca963x:{red,green,blue}[_0.._3]/brightness   (0..255)
  driver: leds-pca963x (NXP pca9635, I2C-13 @ 0x22, 12 通道 = 4 组 RGB)

本实现只封装这条已验证的 sysfs 通路, 硬件访问失败统一抛 HardwareError。
"""

from pathlib import Path

from runtime.errors import HardwareError

LED_SYSFS = Path("/sys/class/leds")
PREFIX = "pca963x:"
COLORS = ("red", "green", "blue")
MAX_BRIGHTNESS = 255


def _channels(color):
    """返回某颜色的全部 channel 名 (如 ['pca963x:red', 'pca963x:red_1', ...])。"""
    return sorted(
        p.name for p in LED_SYSFS.iterdir()
        if p.name.startswith(PREFIX + color)
    )


def _all_channels():
    return sorted(
        p.name for p in LED_SYSFS.iterdir()
        if p.name.startswith(PREFIX)
    )


def _write(channel, value):
    try:
        (LED_SYSFS / channel / "brightness").write_text(str(value))
    except OSError as e:
        raise HardwareError(
            f"写 LED 失败 {channel}={value}: {e!r}",
            capability="led", details={"channel": channel, "value": value},
        ) from e


def _read(channel):
    try:
        return int((LED_SYSFS / channel / "brightness").read_text().strip())
    except (OSError, ValueError) as e:
        raise HardwareError(
            f"读 LED 失败 {channel}: {e!r}",
            capability="led", details={"channel": channel},
        ) from e


def _require_channels(color):
    chs = _channels(color)
    if not chs:
        raise HardwareError(
            f"未找到 LED 通道: {color}", capability="led", details={"color": color})
    return chs


def set_color(args):
    """设置 4 颗 LED 的某一颜色通道亮度。"""
    color, brightness = args["color"], args["brightness"]
    channels = _require_channels(color)
    for ch in channels:
        _write(ch, brightness)
    return {
        "color": color,
        "brightness": brightness,
        "channels": channels,
        "count": len(channels),
    }


def set_rgb(args):
    """把 4 颗 LED 全部设为同一 RGB 颜色。"""
    channels = []
    for color in COLORS:
        chs = _require_channels(color)
        for ch in chs:
            _write(ch, args[color])
        channels.extend(chs)
    return {"channels": channels, "count": len(channels)}


def off(args=None):
    """熄灭全部 LED。"""
    channels = []
    for color in COLORS:
        for ch in _channels(color):
            _write(ch, 0)
            channels.append(ch)
    return {"channels": channels, "count": len(channels), "all_off": True}


def list_channels(args=None):
    """(Query 型示例) 列举全部 LED 通道与当前亮度。"""
    channels = _all_channels()
    return {
        "channels": channels,
        "count": len(channels),
        "max_brightness": MAX_BRIGHTNESS,
        "brightness": {ch: _read(ch) for ch in channels},
    }


# action 名 -> 实现函数 (由 capability/led/__init__.py 注册进 Registry)
ACTIONS = {
    "led.set_color": set_color,
    "led.set_rgb": set_rgb,
    "led.off": off,
    "led.list": list_channels,
}
