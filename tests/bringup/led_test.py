#!/usr/bin/env python3
"""VENTUNO Q 板载 RGB LED (pca9635) bring-up 测试.

验证 /sys/class/leds/pca963x:* 的 sysfs 写通路:
  1. 枚举 12 个 channel (红/绿/蓝 × 4)
  2. 依次点亮 红→绿→蓝→白 → 灭, 每次写入后读回校验
  3. 结束后恢复初始状态(全灭)

注意: 本脚本只能验证 sysfs 写通路(软件层)。SSH 无法目视确认 LED 物理点亮,
      物理点亮需人眼看板子确认 → 故软件层 PASS 但"物理确认"需人工。
"""
import time
from pathlib import Path

LEDS_DIR = Path("/sys/class/leds")
COLORS = ("red", "green", "blue")


def names(color):
    return sorted(p.name for p in LEDS_DIR.iterdir()
                  if p.name.startswith(f"pca963x:{color}"))


def read(n, attr="brightness"):
    return (LEDS_DIR / n / attr).read_text().strip()


def write(n, val):
    (LEDS_DIR / n / "brightness").write_text(str(val))
    return read(n)


def set_color(color, val):
    for n in names(color):
        write(n, val)


def verify(color, expect):
    ok = True
    for n in names(color):
        actual = read(n)
        good = (actual == str(expect))
        ok &= good
        print(f"    {n}: 期望={expect} 实际={actual} {'OK' if good else 'FAIL'}")
    return ok


def main():
    print("=== VENTUNO Q 板载 RGB LED (pca9635) bring-up ===")
    alln = sorted(p.name for p in LEDS_DIR.iterdir()
                  if p.name.startswith("pca963x:"))
    print(f"发现 {len(alln)} channel, max_brightness={read(alln[0], 'max_brightness')}")
    initial = {n: read(n) for n in alln}
    print("初始亮度:", initial)

    seq = [
        ("红 RED",   {"red": 255}),
        ("绿 GREEN", {"green": 255}),
        ("蓝 BLUE",  {"blue": 255}),
        ("白 WHITE", {"red": 255, "green": 255, "blue": 255}),
        ("灭 OFF",   {}),
    ]
    overall = True
    for label, target in seq:
        for c in COLORS:
            set_color(c, target.get(c, 0))
        print(f"\n[{label}]")
        for c in COLORS:
            overall &= verify(c, target.get(c, 0))
        time.sleep(0.5)

    # 恢复初始状态
    for n, v in initial.items():
        write(n, v)
    print("\n=== 结果 ===")
    print("sysfs 写通路:", "PASS" if overall else "FAIL")
    print("注意: 物理点亮需人眼确认(SSH 无法目视)。")


if __name__ == "__main__":
    main()
