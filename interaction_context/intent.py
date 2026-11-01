"""Small deterministic adapter for an explicit, temporary posture snooze."""
import re

from task_orchestrator.intent import _chinese_integer

BARE_SNOOZE = {"先别提醒我", "暂时别提醒我", "稍后再提醒", "稍后再提醒我", "晚点再提醒我"}


def posture_snooze_seconds(text):
    text = text.strip(" 。！？!?,，")
    if text in BARE_SNOOZE:
        return 300
    if "坐姿" not in text:
        return None
    if not any(word in text for word in ("别提醒", "不要提醒", "暂停提醒", "提醒暂停", "暂缓")):
        return None
    match = re.search(r"([0-9一二两三四五六七八九十]+)\s*分钟", text)
    minutes = _chinese_integer(match.group(1)) if match else 5
    if not 1 <= minutes <= 60:
        raise ValueError("坐姿提醒暂缓支持1到60分钟")
    return minutes * 60


def explicit_posture_snooze_seconds(text):
    """Recognize one complete temporary control, never a conditional/compound.

    The legacy helper remains available to existing callers. This stricter
    boundary accepts only minutes so other units cannot become a default snooze.
    """
    if not isinstance(text, str):
        return None
    value = text.strip(" 。！？!?,，")
    if value in BARE_SNOOZE:
        return 300
    duration = r"(?:\d{1,3}|[一二两三四五六七八九十]+)\s*分钟"
    prefix = r"(?:请|先|暂时)?"
    grammar = (
        prefix + r"(?:暂停|暂缓)坐姿提醒(?:" + duration + r")?|"
        + prefix + r"(?:坐姿)?(?:先|暂时)?(?:别提醒|不要提醒|暂停提醒|提醒暂停|暂缓)(?:我)?"
        r"(?:坐姿)?(?:" + duration + r")?|"
        r"(?:接下来)?" + duration + r"(?:内)?(?:暂停|暂缓)坐姿提醒|"
        r"(?:接下来)?" + duration + r"(?:内)?(?:先)?(?:别|不要)提醒我坐姿"
    )
    if "坐姿" not in value or not re.fullmatch(grammar, value):
        return None
    match = re.search(r"([0-9一二两三四五六七八九十]+)\s*分钟", value)
    minutes = _chinese_integer(match.group(1)) if match else 5
    if not 1 <= minutes <= 60:
        raise ValueError("坐姿提醒暂缓支持1到60分钟")
    return minutes * 60
