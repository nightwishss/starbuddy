"""Cheap, bounded routing/grounding for requests about current perception.

This selects existing context, not actions or model control loops. A cloud-chat
preference must not silently turn a sensory question into an ungrounded chat.
"""

from __future__ import annotations

import re


def _asks_current_affect(value: str) -> bool:
    # Route explicit current-state questions, not the user's own feelings or a
    # quoted historical question. Clause boundaries keep "how can you help?"
    # from turning a preceding self-report into a request for camera inference.
    past = (r"昨天|前天|昨晚|刚才|之前|以前|上次|过去|曾经|当时|那时|"
            r"\b(?:yesterday|earlier|previously|before|used to|last (?:night|time|week))\b")
    if re.search(past, value) and not re.search(r"现在|此刻|当前|目前|\b(?:now|currently)\b", value):
        return False
    for clause in re.split(r"[,，;；。.!！?？\n]", value):
        if (re.search(past, clause) or (re.search(past, value)
                and re.search(r"你(?:问|说)|你曾问|\b(?:you asked|you said)\b", clause))):
            continue
        self_state = re.search(r"我.{0,10}(?:状态|心情|情绪|表情|看起来)|(?:我的?)?表情", clause)
        if self_state and re.search(
            r"(?:状态|心情|情绪|表情|看起来).{0,8}(?:如何|怎么样|怎样|咋样|是什么)|"
            r"我.{0,8}(?:什么|哪种).{0,4}(?:表情|心情|情绪|状态)|"
            r"(?:表情|心情|情绪|状态).{0,4}(?:吗|么)\s*$|"
            r"(?:看看|看一看|观察|判断|识别).{0,8}(?:我|表情)", clause
        ):
            return True
        if re.search(
            r"\bhow (?:do i look|am i looking)\b|"
            r"\b(?:what(?:'s| is)|how(?:'s| is)|is) my (?:mood|(?:facial )?expression)\b|"
            r"\b(?:what|how) does my (?:facial )?expression\b|"
            r"\b(?:tell|describe|read|assess|check|look at) (?:me )?(?:what )?(?:my (?:mood|(?:facial )?expression)|how i look)\b",
            clause,
        ):
            return True
    return False


def perception_query(text: str) -> str:
    """Return a semantic context domain; no model invocation or side effects."""
    value = text.lower().strip()
    if re.search(r"我是谁|我叫什么|认识我|认出我|记得我|知道我是谁|知道我的名字|"
                 r"\bwho am i\b|\b(?:recognize|recognise|remember|know) me\b|"
                 r"\bwhat(?:'s| is) my name\b", value):
        return "identity"
    # Metaphorical/textual uses of 'see' do not need a camera refresh.
    if re.search(r"(?:看|检查).{0,5}(?:代码|文章|文档|链接)|"
                 r"\bsee (?:my|the) (?:code|point|message|document)\b", value):
        return "chat"
    if re.search(r"我(?:现在|目前)?在(?:哪里|哪儿|哪边|什么位置)|"
                 r"我.{0,8}(?:画面|镜头).{0,6}(?:位置|哪|左|右)|"
                 r"\bwhere am i\b|\bwhere (?:can you|do you) see me\b", value):
        return "location"
    if re.search(r"坐姿|体态|驼背|\bposture\b|\bslouch", value):
        return "posture"
    if _asks_current_affect(value):
        return "affect"
    if re.search(r"(?:看|瞧)(?:得|的|不)?(?:到|见)(?:了)?(?:我|你|人)|看看我|"
                 r"(?:能|可以|能不能).{0,4}看(?:到|见)(?:吗|么)|"
                 r"(?:有|没).{0,3}人在|\b(?:see|seeing) (?:me|us|anyone|anybody|my face)\b|"
                 r"\bam i (?:visible|in (?:the )?(?:frame|view))\b", value):
        return "presence"
    if re.search(r"看到什么|看见什么|周围有什么|谁在这里|感知信息|周围物体|几个人|多少人|"
                 r"\bhow many (?:people|persons|faces)\b|"
                 r"镜头|画面|摄像头|\b(?:what|who).{0,20}(?:see|camera|frame)\b", value):
        return "scene"
    return "chat"


def has_person(context: dict) -> bool:
    perception = context.get("perception") or {}
    return bool(perception.get("people") or any(
        o.get("label") == "person" and o.get("count", 0) > 0
        for o in perception.get("objects", [])))


def presence_answer(context: dict) -> str:
    """Grounding fallback, not a substitute for the ordinary conversation LLM."""
    if has_person(context):
        if context.get("identity_verified"):
            return f"镜头里检测到了人，当前已授权身份识别结果是{context['identity_label']}。"
        return "摄像头画面里检测到了人，不过还不能确认是不是正在和我说话的你。"
    state = (context.get("evidence", {}).get("people") or {}).get("state")
    reasons = {
        "observed_absent": "最新画面暂时没有检测到人脸，这不等于画面中一定没人。",
        "stale": "视觉观测已经过期，现在不能据此确认有没有看到你。",
        "unavailable": "当前视觉数据源暂不可用，所以还不能确认有没有看到你。",
        "disabled": "当前相关感知没有开启，所以还不能确认有没有看到你。",
    }
    return reasons.get(state, "当前没有足够可靠的视觉观测，还不能确认有没有看到你。")


def contradicts_presence(answer: str, context: dict) -> bool:
    """Reject capability denial/personal attribution inconsistent with evidence."""
    value = answer.lower()
    detected = has_person(context)
    denial = bool(re.search(
        r"(?:看不到|看不见|没(?:有)?看到|无法看到|不能看到)(?:你|任何人|人|画面)|"
        r"没有摄像头|"
        r"(?:没有|不具备)(?:任何)?(?:视觉|摄像头|看见).*?(?:功能|能力)|"
        r"(?:不能|无法).{0,8}(?:访问|获取|查看|看到).{0,8}(?:摄像头|画面|图像)|"
        r"\b(?:cannot|can't|unable to|don't|do not) (?:see|access|view)\b", value))
    personal_claim = bool(re.search(
        r"(?<!没)看到你|(?<!没)看见你|看得[到见]你|\b(?:i can see you|i see you)\b", value))
    if detected and denial:
        return True
    if personal_claim and not context.get("identity_verified"):
        return True
    return not detected and bool(re.search(r"(?:检测到|确认有|看到了?).{0,4}(?:人|你)|"
                                          r"\bi (?:can )?see (?:a person|someone|you)\b", value))
