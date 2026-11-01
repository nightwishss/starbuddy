"""Local visual-identity replies, not speaker identification or profile access."""
import re

from interaction_context.turn import assemble_turn_context


def is_face_question(message):
    # Normalize a temporary ROUTING copy only: original ASR/history and local
    # enrolled labels must not be converted. Each accepted clause is still a
    # complete identity question; never search a substring in another topic.
    from unicodedata import normalize
    if not isinstance(message, str) or not 0 < len(message) <= 300:
        return False
    value = normalize('NFKC', message).lower().translate(str.maketrans({
        '認': '认', '識': '识', '記': '记', '誰': '谁', '嗎': '吗', '麼': '么',
        '鏡': '镜', '頭': '头', '攝': '摄', '裡': '里', '這': '这', '個': '个',
        '請': '请', '問': '问', '還': '还', '現': '现', '夠': '够', '會': '会',
        '’': "'",
    }))
    value = re.sub(r'\s+', ' ', value).strip()
    value = re.sub(r'(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])', '', value)
    separator = r'[\s,，。.!！？?;；:：…]*'
    salutation = r'(?:你好|您好|嗨|芽芽|星伴|hello|hi)'
    prefix = rf'(?:{salutation}{separator}){{0,2}}'
    modal = r'(?:还|现在|这次)?(?:能不能|可不可以|会不会|能够|可以|能|会|是否|能否)?'
    particle = r'(?:吗|么|嘛|吧|呀|呢|啊)?'
    recognition = r'(?:认不认识|认不认得(?:出)?|记不记得|认识|认出|认得(?:出)?|记得)'
    know = r'(?:知不知道|知道不知道|知道)'
    question = (
        r'(?:请问)?(?:你)?' + modal + recognition + r'我(?:了|是谁)?' + particle + r'|'
        r'(?:请问)?(?:你)?' + modal + r'(?:' + know + r')?'
            r'(?:我(?:是谁|叫什么(?:名字)?)|我的名字(?:是什么)?)' + particle + r'|'
        r'(?:请问)?(?:你)?(?:现在)?(?:镜头|画面|摄像头)里'
            r'(?:认出的是|看到的是|认出|看到|有|的|是)?(?:谁|哪位)' + particle + r'|'
        r'(?:你)?(?:认出|认得|识别出)(?:的是)?(?:谁|哪位)(?:了)?' + particle + r'|'
        r'who am i|what(?:\x27s| is) my name|'
        r'(?:(?:do|can|could) you )?(?:recognize|recognise|remember|know) me'
    )
    clause = prefix + '(?:' + question + ')'
    # ASR can omit sentence punctuation: "吗你知道..." remains two identity
    # clauses. An author/friend question, quoted history, another task, or a
    # fourth clause leaves unmatched content and stays on ordinary dialogue.
    return re.fullmatch(clause + rf'(?:{separator}{clause}){{0,2}}' + separator, value) is not None


def visual_identity_reply(message, snapshot, identity, *, now=None, requested=False):
    """Use only a currently consented, uniquely associated local face label.

    The synthetic text input below admits VISUAL identity, never proves that
    the microphone speaker/touch operator is that person. No profile is read.
    The returned reply must not enter the cloud dialogue history.
    """
    if requested is not True and not is_face_question(message):
        return None
    context = assemble_turn_context({'input_type': 'text'}, snapshot, identity, [], now=now)
    label = context.get('identity_label') if context.get('identity_verified') else None
    front_count = len([p for p in context['perception']['people'] if p.get('source') == 'camera.front'])
    if label:
        evidence_state = 'recognized'
        response = f'刚才镜头里，我认出的是{label}。你好！我们可以接着聊。'
    elif front_count > 1:
        evidence_state = 'multiple_faces'
        response = '镜头里有不止一张脸，我先不选一个称呼。我们照样可以聊天。'
    else:
        evidence_state = 'unconfirmed'
        response = '这次还没认出镜头里的人。可以先和我聊天，不用等识别。'
    return {'intent': 'chat', 'response': response, 'backend': 'local-face-context',
            'model_calls': 0, 'private_context': True,
            'identity_evidence': {'state': evidence_state, 'front_count': front_count}}
