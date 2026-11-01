"""GPIO Capability 实现 (Query 型, MPU backend, 只读) — Capability Contract v0.1.

底层真实接口: libgpiod CLI (gpiodetect / gpioinfo / gpioget)。
  板子无 pip / python gpiod 模块, 故用 subprocess 调 CLI。
Phase 1 bring-up 已实测: 4 个 gpiochip (0/1=PMIC, 2/3=SoC pinctrl), 共 181 line。

只读: 枚举 chip / 枚举 line / 读电平。
⚠️ 输出写 (gpioset) 未开放 —— 无外设/原理图, 冒然驱动输出可能碰到内部信号;
   待拿到原理图或接入外设确认安全 line 后再加 .mcu(UNO盾/Bridge) 与写路径。
"""
import subprocess

from runtime.errors import HardwareError


def _run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise HardwareError(
            f"libgpiod 命令失败 {cmd!r}: {r.stderr.strip()}",
            capability="gpio", details={"cmd": cmd})
    return r.stdout


def _parse_chips():
    chips = []
    for ln in _run("gpiodetect").splitlines():
        if "[" not in ln:
            continue
        name = ln.split("[")[0].strip()
        label = ln.split("[")[1].split("]")[0]
        num_lines = 0
        if "(" in ln:
            try:
                num_lines = int(ln.split("(")[1].split()[0])
            except (ValueError, IndexError):
                num_lines = 0
        chips.append({"name": name, "label": label, "num_lines": num_lines})
    return chips


def list_chips(args=None):
    chips = _parse_chips()
    return {"chips": chips, "count": len(chips)}


def list_lines(args):
    chip = args["chip"]
    lines = [ln.strip() for ln in _run(f"gpioinfo {chip}").splitlines()
             if ln.strip().startswith("line")]
    return {"chip": chip, "count": len(lines), "lines": lines}


def read(args):
    chip, offset = args["chip"], args["offset"]
    out = _run(f"gpioget {chip} {offset}").strip()
    try:
        value = int(out)
    except ValueError:
        value = None
    return {"chip": chip, "offset": offset, "value": value}


# action 名 -> 实现函数 (由 capability/gpio/__init__.py 注册进 Registry)
ACTIONS = {
    "gpio.list_chips": list_chips,
    "gpio.list_lines": list_lines,
    "gpio.read": read,
}
