#!/usr/bin/env python3
"""VENTUNO Q MPU GPIO bring-up 测试 (只读).

验证 libgpiod 通路, 不驱动任何输出 (无外设/无原理图, 冒然驱动输出可能碰到内部信号):
  1. 枚举 4 个 gpiochip 共 181 条 line (gpioinfo)
  2. 区分 used (已被内核占用) / unused line
  3. 对 unused input line 用 gpioget 读值 → 验证读通路
  4. 对 used line (perst/interrupt) 用 gpioget 读 → 应报 busy, 证明内核占用正常

运行: python3 tests/bringup/gpio_test.py
"""
import subprocess
import sys


def run(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def parse_gpioinfo():
    """返回 {chip名: [line 记录...]}，line 记录含 offset/name/consumer/direction。"""
    det = run("gpiodetect").stdout
    chips = {}
    for ln in det.splitlines():
        if "[" in ln and "(" in ln:
            name = ln.split("[")[0].strip()          # gpiochip0
            chips[name] = {"label": ln.split("[")[1].split("]")[0], "lines": []}

    info = run("gpioinfo").stdout
    cur = None
    for ln in info.splitlines():
        if ln.startswith("gpiochip"):
            cur = ln.split(" ")[0].rstrip(":")
            chips.setdefault(cur, {"label": "", "lines": []})
        elif ln.strip().startswith("line") and cur:
            chips[cur]["lines"].append(ln.strip())
    return chips


def main():
    print("=== VENTUNO Q MPU GPIO bring-up (只读) ===")
    chips = parse_gpioinfo()
    results = []

    def check(cond, label):
        results.append(bool(cond))
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    print(f"\n枚举到 {len(chips)} 个 gpiochip")
    check(len(chips) == 4, "4 个 gpiochip")

    total_lines = 0
    used_lines = []
    unused_input = {}
    for name, c in chips.items():
        n = len(c["lines"])
        total_lines += n
        print(f"  {name} [{c['label']}]: {n} lines")
        for ln in c["lines"]:
            # 形如: line  2: unnamed "perst" output active-low [used]
            offset = int(ln.split("line")[1].split(":")[0].strip())
            is_used = "[used]" in ln
            if is_used:
                used_lines.append((name, offset, ln))
            else:
                # 记录 unused 且 input 方向的 line (用于 gpioget 读)
                if "input" in ln:
                    unused_input.setdefault(name, []).append(offset)
    check(total_lines == 181, f"总 line 数 = {total_lines} (期望 181 = 12+12+134+23)")

    print(f"\n[used lines] 共 {len(used_lines)} 条 (内核已占用, 不碰):")
    for name, off, ln in used_lines:
        print(f"  {name} line {off}: {ln}")

    # 读通路: gpioget 每个 chip 第一条 unused input line
    print("\n[gpioget 读通路验证]")
    read_ok = 0
    for name in chips:
        cands = unused_input.get(name, [])
        if not cands:
            print(f"  {name}: 无 unused input line 可读")
            continue
        off = cands[0]
        r = run(f"gpioget {name} {off}")
        if r.returncode == 0:
            read_ok += 1
            print(f"  {name} line {off}: 值={r.stdout.strip()} (读通路 OK)")
        else:
            print(f"  {name} line {off}: gpioget 失败 -> {r.stderr.strip()}")
    check(read_ok >= 1, f"gpioget 读通路正常 (成功读 {read_ok} 个 chip)")

    # used line 占用验证: 读 perst/interrupt 应报 busy
    print("\n[used line 占用验证]")
    busy_ok = False
    for name, off, ln in used_lines:
        r = run(f"gpioget {name} {off}")
        if "busy" in r.stderr.lower() or "busy" in r.stdout.lower():
            busy_ok = True
            print(f"  {name} line {off}: 正确返回 busy (内核已占用)")
            break
        elif r.returncode != 0:
            print(f"  {name} line {off}: 返回错误 {r.stderr.strip()}")
    check(busy_ok, "used line 被内核正确占用 (gpioget 报 busy)")

    total = len(results)
    passed = sum(results)
    print(f"\n=== 结果: {passed}/{total} PASS ===")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
