#!/usr/bin/env python3
"""VENTUNO Q 输入设备 bring-up 测试 (只读, 不按压按键).

验证输入子系统通路:
  1. 枚举 /dev/input/event* 与 /proc/bus/input/devices
  2. 打开每个 event 设备, 读设备名 (EVIOCGNAME)
  3. 读支持的 EV 类型位 (EVIOCGBIT)

⚠️ 不做物理按压测试: 板子只有 pmic_pwrkey(电源键) / pmic_resin(复位键),
   按压复位键会热重启(断 SSH), 按压电源键可能触发关机。

运行: python3 tests/bringup/input_test.py
"""
import fcntl
import os
import struct
import sys
from pathlib import Path


def _IOC(direction, type_, nr, size):
    return (direction << 30) | (ord(type_) << 8) | nr | (size << 16)


EVIOCGNAME = _IOC(2, "E", 0x06, 256)     # 读设备名
EVIOCGBIT_EV = _IOC(2, "E", 0x20, 64)    # 读 EV 类型位图 (64 字节 = 512 位)

EV_NAMES = {
    0x00: "SYN", 0x01: "KEY", 0x02: "REL", 0x03: "ABS",
    0x04: "MSC", 0x11: "LED", 0x12: "SND", 0x14: "SW",
}


def read_name(fd):
    try:
        buf = fcntl.ioctl(fd, EVIOCGNAME, bytes(256))
        return buf.split(b"\x00")[0].decode("utf-8", "replace")
    except OSError as e:
        return f"<ioctl失败: {e}>"


def read_ev_types(fd):
    try:
        buf = fcntl.ioctl(fd, EVIOCGBIT_EV, bytes(64))
        bits = set()
        for byte_i, b in enumerate(buf):
            for bit_i in range(8):
                if b & (1 << bit_i):
                    bits.add(byte_i * 8 + bit_i)
        return sorted(bits)
    except OSError as e:
        return f"<ioctl失败: {e}>"


def main():
    print("=== VENTUNO Q 输入设备 bring-up (只读) ===")
    results = []

    def check(cond, label):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    # 1. /proc/bus/input/devices 里的设备
    proc = Path("/proc/bus/input/devices")
    dev_lines = []
    if proc.exists():
        dev_lines = [l for l in proc.read_text().splitlines() if l.startswith("N: Name=")]
    print("\n[输入设备清单]")
    for l in dev_lines:
        print("  " + l)
    check(len(dev_lines) == 2, f"识别到 {len(dev_lines)} 个输入设备 (期望 2: pwrkey/resin)")

    # 2. 枚举 /dev/input/event*
    events = sorted(Path("/dev/input").glob("event*"))
    print(f"\n[/dev/input/event*] 共 {len(events)} 个")
    opened = 0
    for ev in events:
        try:
            fd = os.open(str(ev), os.O_RDONLY | os.O_NONBLOCK)
            name = read_name(fd)
            ev_types = read_ev_types(fd)
            os.close(fd)
            opened += 1
            ev_s = ", ".join(EV_NAMES.get(t, f"0x{t:02x}") for t in ev_types)
            print(f"  {ev.name}: name={name!r}  EV类型=[{ev_s}]")
        except OSError as e:
            print(f"  {ev.name}: 打开失败 {e}")
    check(opened == len(events), f"全部 {opened}/{len(events)} 个 event 设备可打开")

    # 3. 权限检查 (arduino 应在 input 组, event 组可读)
    import stat
    readable = all(ev.stat().st_mode & stat.S_IRGRP for ev in events)
    check(readable, "event 设备组可读 (arduino 在 input 组)")

    total = len(results)
    passed = sum(results)
    print(f"\n=== 结果: {passed}/{total} PASS ===")
    print("注: 物理按键按压测试未做 (复位键会重启, 电源键可能关机)。")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
