"""One-call text dialogue / read-only local-context proposal protocol.

This module never reads a camera, identity, profile, credential or device. A
proposal conveys no authorization and contains no arguments or identity label.
"""
from __future__ import annotations

import json
import re

from gateway.speech.text import INCOMPLETE_CHAT_REPLY, public_chat_text


DIALOGUE_CONTEXT_RULES = (
    '\n【当前对话的只读情境协议 v1】'
    '你是正在与现场用户互动的芽芽桌面伙伴的对话部分，不是一个与设备无关的纯文字聊天页。'
    '这台设备已安装本地人脸识别服务，宿主可以读取当前可见人脸的已授权身份线索；'
    '你不直接接收画面和姓名，但可以用下面的只读请求让宿主在本地核验。'
    '输入没有姓名只说明姓名未交给文字模型，不是本地识别未知的证据；是否认出由宿主核验。'
    '普通聊天继续直接输出自然回答，不输出JSON。'
    '结合本轮原话和最近按角色区分的问答理解意图，不要求用户使用固定句式；'
    '连续追问、口语、简繁体或重复提问也按其意思理解。'
    '在这次现场互动中，用户询问你认不认得或记不记得当前的我、我是谁、'
    '能否叫出我的名字或如何称呼我，通常是在请设备核实当前可见的人脸身份；'
    '即使没说镜头或摄像头，也优先向宿主请求核验，而不是作纯文字身份猜测。'
    '仅问记不记得当前来访者，不等于要求回忆过去发生的事情。'
    '询问当前镜头里的人是谁，以及承接这种身份话题的指代追问，也请求本地核验。'
    '不要猜名字，也不要因为本提示没有姓名就断言不认识，'
    '不要用我是语言模型、没有视觉能力或没有长期记忆来否定设备已有的本地核验能力；'
    '只输出一个完整JSON对象：{"context_request":"visible_identity"}。'
    '该对象不能有其他字段、姓名、标签、参数、解释、Markdown或附加文本。'
    '它只是请宿主在本地读取已授权的新鲜视觉线索，不是识别结果或操作授权。'
    '但用户明确询问朋友/作者等第三人的知识、引用过去的问题、'
    '要求回忆上次的对话或过去发生的具体内容、读取个人档案、'
    '作自我介绍或确认是谁正在说话，不等于查询当前可见人脸身份；'
    '这些情况正常回答或自然澄清，不发出该视觉请求。'
    '镜头中出现的人不证明谁在说话或触屏。'
    '如果同时有其他实质任务，不用身份JSON取代整段需求；自然回应或先问清。'
    '意图示例：当前来访者问“能叫出我的名字不？”→{"context_request":"visible_identity"}；'
    '最近在询问当前来访者的视觉身份，接着问“那该怎么称呼我呢？”→{"context_request":"visible_identity"}。'
    '相反，“还记得我们上回聊的故事吗”是历史内容问题，“你认识写这本书的作者吗”是第三人问题，'
    '“一起画朵花吧”是普通互动，都用自然文字回应，不输出身份请求。'
    '你不会收到实际视觉身份结果，不能自行生成结果、读取档案、执行工具或设备操作。'
    '只输出最终自然回答或上述单字段JSON，不输出思考过程。'
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate dialogue context field')
        result[key] = value
    return result


def parse_dialogue_reply(raw):
    """Return one standard proposal, never a spoken JSON/control envelope.

    Ordinary text (including a bare numeric answer) stays ordinary text. Any
    structured reply must be exactly the sole supported argument-free request;
    partial, wrapped, duplicate or forged request objects fail without inference.
    """
    if type(raw) is not str or not 0 < len(raw.strip()) <= 500:
        raise ValueError('invalid bounded dialogue reply')
    text = public_chat_text(raw)
    if text == INCOMPLETE_CHAT_REPLY:
        raise ValueError('incomplete dialogue reply')
    structured = (text.startswith(('{', '[', '```'))
                  or re.search(r'\bcontext_request["\x27]?\s*[:=]', text)
                  or text in {'context_request', 'visible_identity'})
    if structured:
        try:
            value = json.loads(text, object_pairs_hook=_unique_object)
        except (ValueError, TypeError) as exc:
            raise ValueError('invalid dialogue context request') from exc
        if (type(value) is not dict or set(value) != {'context_request'}
                or value['context_request'] != 'visible_identity'):
            raise ValueError('unsupported dialogue context request')
        return {'intent': 'chat', 'response': '',
                'requested_capability': 'perception.visible_identity'}
    return {'intent': 'chat', 'response': text.strip(), 'requested_capability': None}
