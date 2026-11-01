"""Bounded, exact arithmetic with a single ephemeral verified-result receipt.

No eval, model, file, network, device operation, or prose-derived arithmetic.
prepare() proposes only; commit() follows host authority/output acceptance.
"""
from __future__ import annotations

from decimal import Decimal, DecimalException, Inexact, localcontext
import hashlib
import hmac
import json
import math
import re
import secrets
import threading
import time


BACKEND = 'verified-local-arithmetic'
MAX_ABS = Decimal('1000000000')
MAX_OPERATIONS = 8
MAX_DIGITS = 24
MAX_MESSAGE = 320
_CN_DIGITS = {'零': 0, '〇': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
              '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
_NUMBER = r'(?:\d+(?:\.\d+)?|\.\d+|[零〇一二两三四五六七八九十百千万点负]+)'
_OPERATOR = r'(?:divided\s+by|plus|minus|times|加上?|减去?|乘以?|除以?|[+＋×÷*/−-])'
_TOKEN = re.compile(r'\s*(' + _NUMBER + '|' + _OPERATOR + r'|[()（）])', re.I)
_FOLLOWUP = re.compile(r'(?:那|那么)?\s*(?:再|然后|接着)\s*(' + _OPERATOR + r')\s*(' + _NUMBER + r')', re.I)
_SIGNED_FIELDS = {'answer', 'result', 'backend', 'error', '_generation', '_issued_at', '_nonce'}
_PENDING_FIELDS = _SIGNED_FIELDS | {'_seal'}


class _ArithmeticError(ValueError):
    pass


def _chinese_group(value):
    if value == 0:
        return '零'
    result, gap = '', False
    for divisor, unit in ((1000, '千'), (100, '百'), (10, '十'), (1, '')):
        digit, value = divmod(value, divisor)
        if digit:
            if gap:
                result += '零'
            result += ('' if divisor == 10 and digit == 1 and not result else '零一二三四五六七八九'[digit]) + unit
            gap = False
        elif result and value:
            gap = True
    return result


def _parse_group(value):
    if not value:
        raise ValueError('missing_chinese_number')
    if all(character in _CN_DIGITS for character in value):
        return int(''.join(str(_CN_DIGITS[character]) for character in value))
    clean = value.replace('零', '')
    digit = '[一二三四五六七八九]'
    match = re.fullmatch(r'(?:(' + digit + r')千)?(?:(' + digit + r')百)?(?:(' + digit + r')?十)?(' + digit + r')?', clean)
    if not match or not clean:
        raise ValueError('invalid_chinese_number')
    thousands, hundreds, tens, ones = match.groups()
    number = sum((_CN_DIGITS[part] if part else 0) * scale
                 for part, scale in ((thousands, 1000), (hundreds, 100), (tens, 10), (ones, 1)))
    if '十' in clean and tens is None:
        number += 10
    canonical = _chinese_group(number)
    if value != canonical and not (value.startswith('一十') and value[1:] == canonical):
        raise ValueError('ambiguous_chinese_number')
    return number


def _number(value):
    if len(value) > 40:
        raise _ArithmeticError('limit')
    if re.fullmatch(r'(?:\d+(?:\.\d+)?|\.\d+)', value):
        if sum(character.isdigit() for character in value) > MAX_DIGITS:
            raise _ArithmeticError('limit')
        return _bounded(Decimal(value))
    negative = value.startswith('负')
    value = value[1:] if negative else value
    value = value.replace('〇', '零').replace('两', '二')
    if not value or value.count('点') > 1 or value.count('万') > 1:
        raise ValueError('invalid_chinese_number')
    integer, separator, fraction = value.partition('点')
    if separator and (not fraction or not all(character in _CN_DIGITS for character in fraction)):
        raise ValueError('invalid_chinese_decimal')
    if '万' in integer:
        high, low = integer.split('万')
        if not high:
            raise ValueError('invalid_chinese_number')
        high_number = _parse_group(high)
        low_number = _parse_group(low.lstrip('零')) if low.lstrip('零') else 0
        if not 1 <= high_number <= 9999 or not 0 <= low_number <= 9999:
            raise _ArithmeticError('limit')
        number = high_number * 10000 + low_number
        canonical = _chinese_group(high_number) + '万'
        if low_number:
            canonical += ('零' if low_number < 1000 else '') + _chinese_group(low_number)
        if integer != canonical:
            raise ValueError('ambiguous_chinese_number')
    else:
        number = _parse_group(integer) if integer else 0
    raw = str(number) + ('.' + ''.join(str(_CN_DIGITS[character]) for character in fraction) if separator else '')
    if sum(character.isdigit() for character in raw) > MAX_DIGITS:
        raise _ArithmeticError('limit')
    return _bounded(Decimal(('-' if negative else '') + raw))


def _format(value):
    if value == 0:
        return '0'
    raw = format(value, 'f')
    return raw.rstrip('0').rstrip('.') if '.' in raw else raw


def _bounded(value):
    if not value.is_finite() or abs(value) > MAX_ABS:
        raise _ArithmeticError('limit')
    canonical = _format(value)
    significant = canonical.lstrip('-').replace('.', '').lstrip('0')
    if len(significant) > MAX_DIGITS:
        raise _ArithmeticError('limit')
    return value


def _operator(value):
    value = re.sub(r'\s+', ' ', value.lower())
    for names, operator in ((('+', '＋', '加', '加上', 'plus'), '+'),
                             (('-', '−', '减', '减去', 'minus'), '-'),
                             (('*', '×', '乘', '乘以', 'times'), '*'),
                             (('/', '÷', '除', '除以', 'divided by'), '/')):
        if value in names:
            return operator
    return value


def _apply(left, operator, right):
    if operator == '+':
        result = left + right
    elif operator == '-':
        result = left - right
    elif operator == '*':
        result = left * right
    elif operator == '/':
        if right == 0:
            raise _ArithmeticError('division_by_zero')
        result = left / right
    else:
        raise ValueError('invalid_operator')
    return _bounded(result)


def _clean_message(message):
    value = message.lower().strip().rstrip('？?。.!！').strip()
    value = re.sub(r'(?:呢|吗|啊|呀)\s*$', '', value).strip()
    value = re.sub(r'\s*(?:(?:等于|结果是|是)\s*(?:几|多少|什么)?|=)\s*$', '', value).strip()
    return value


def _expression(message):
    value = re.sub(r'^(?:what\s+is\s+|calculate\s+)', '', message)
    value = re.sub(r'^(?:现在)?(?:(?:请帮我|请问|帮我|请)\s*)?(?:(?:计算|算)(?:一下|算)?\s*)?', '', value).strip()
    tokens, offset = [], 0
    while offset < len(value):
        match = _TOKEN.match(value, offset)
        if not match:
            return None
        tokens.append(match.group(1))
        offset = match.end()
    if not tokens:
        return None
    position, operations = 0, 0

    def factor(depth=0):
        nonlocal position
        if depth > 4:
            raise _ArithmeticError('limit')
        if position >= len(tokens):
            raise ValueError('invalid_expression')
        sign = 1
        if tokens[position] in ('+', '-', '−', '＋'):
            sign = -1 if tokens[position] in ('-', '−') else 1
            position += 1
            if position >= len(tokens):
                raise ValueError('invalid_expression')
        token = tokens[position]
        position += 1
        if token in ('(', '（'):
            value = expression(depth + 1)
            if position >= len(tokens) or tokens[position] not in (')', '）'):
                raise ValueError('invalid_expression')
            position += 1
        else:
            value = _number(token)
        return _bounded(value * sign)

    def product(depth):
        nonlocal position, operations
        value = factor(depth)
        while position < len(tokens) and _operator(tokens[position]) in ('*', '/'):
            operator = _operator(tokens[position]); position += 1; operations += 1
            if operations > MAX_OPERATIONS:
                raise _ArithmeticError('limit')
            value = _apply(value, operator, factor(depth))
        return value

    def expression(depth=0):
        nonlocal position, operations
        value = product(depth)
        while position < len(tokens) and _operator(tokens[position]) in ('+', '-'):
            operator = _operator(tokens[position]); position += 1; operations += 1
            if operations > MAX_OPERATIONS:
                raise _ArithmeticError('limit')
            value = _apply(value, operator, product(depth))
        return value

    try:
        with localcontext() as context:
            context.prec = 28
            context.traps[Inexact] = True
            result = expression()
            if position != len(tokens) or operations == 0:
                return None
            return result
    except DecimalException as error:
        raise _ArithmeticError('precision') from error


class ArithmeticContext:
    """One verified Decimal base, bounded TTL and generation-fenced proposals."""
    def __init__(self, *, clock=time.monotonic, ttl_seconds=900):
        if type(ttl_seconds) not in (int, float) or not math.isfinite(ttl_seconds) or not 0 < ttl_seconds <= 900:
            raise ValueError('invalid_arithmetic_ttl')
        self._clock, self._ttl = clock, float(ttl_seconds)
        self._lock = threading.RLock()
        self._secret = secrets.token_bytes(32)
        self._generation = 0
        self._base = self._expires = None

    def _now(self):
        now = self._clock()
        if type(now) not in (int, float) or not math.isfinite(now):
            raise ValueError('invalid_arithmetic_clock')
        return now

    def _expire(self, now):
        if self._expires is not None and now >= self._expires:
            self._base = self._expires = None
            self._generation += 1

    def _seal(self, value):
        data = json.dumps({key: value[key] for key in sorted(_SIGNED_FIELDS)},
                          ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
        return hmac.new(self._secret, data, hashlib.sha256).hexdigest()

    def _pending(self, result, error, now, *, followup=False):
        answers = {
            'no_base': '请先说一个完整算式，我还没有可接着计算的已确认结果。',
            'division_by_zero': '除数不能为0，请换一个算式。',
            'precision': '这个结果无法在本次精度范围内精确表示，请换一个算式。',
            'limit': '这道算式超出本次可精确计算的范围，请简化后再试。',
        }
        canonical = _format(result) if result is not None else None
        value = {'answer': answers[error] if error else ('接着计算，结果是' if followup else '计算结果是') + canonical + '。',
                 'result': canonical, 'backend': BACKEND, 'error': error,
                 '_generation': self._generation, '_issued_at': now, '_nonce': secrets.token_hex(16)}
        value['_seal'] = self._seal(value)
        return value

    def prepare(self, message):
        if type(message) is not str or not 0 < len(message.strip()) <= MAX_MESSAGE:
            return None
        with self._lock:
            now = self._now(); self._expire(now)
            value = _clean_message(message)
            followup = _FOLLOWUP.fullmatch(value)
            try:
                if followup:
                    right = _number(followup.group(2))
                    if self._base is None:
                        return self._pending(None, 'no_base', now, followup=True)
                    with localcontext() as context:
                        context.prec = 28
                        context.traps[Inexact] = True
                        result = _apply(self._base, _operator(followup.group(1)), right)
                else:
                    result = _expression(value)
                    if result is None:
                        return None
                return self._pending(result, None, now, followup=followup is not None)
            except _ArithmeticError as error:
                return self._pending(None, str(error), now, followup=followup is not None)
            except DecimalException:
                return self._pending(None, 'precision', now, followup=followup is not None)
            except ValueError:
                return None

    def commit(self, pending):
        with self._lock:
            now = self._now(); self._expire(now)
            try:
                if (type(pending) is not dict or set(pending) != _PENDING_FIELDS
                        or type(pending['_generation']) is not int or pending['_generation'] != self._generation
                        or type(pending['_issued_at']) not in (int, float)
                        or not math.isfinite(pending['_issued_at'])
                        or not pending['_issued_at'] <= now < pending['_issued_at'] + self._ttl
                        or type(pending['_seal']) is not str
                        or not hmac.compare_digest(pending['_seal'], self._seal(pending))):
                    return False
                result = _bounded(Decimal(pending['result'])) if pending['result'] is not None else None
            except (ValueError, TypeError, KeyError, DecimalException):
                return False
            self._base = result
            self._expires = now + self._ttl if result is not None else None
            self._generation += 1
            return True

    def invalidate(self):
        with self._lock:
            self._base = self._expires = None
            self._generation += 1
