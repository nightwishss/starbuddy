"""Deterministic text boundaries for ordinary chat and bounded speech chunks."""

from __future__ import annotations

import re


INCOMPLETE_CHAT_REPLY = "这次回答没有生成完整，请再问我一次。"
_REASONING_TAG = re.compile(r"<\s*(/?)\s*(think|analysis|reasoning)\s*>", re.IGNORECASE)
_REASONING_FRAGMENT = re.compile(r"<\s*/?\s*(?:think|analysis|reasoning)\b", re.IGNORECASE)
_PARTIAL_REASONING_TAG = re.compile(
    r"<\s*/?\s*(?:" + "|".join(sorted({name[:length]
        for name in ("think", "analysis", "reasoning")
        for length in range(1, len(name))}, key=len, reverse=True)) + r")?\s*$",
    re.IGNORECASE,
)
_DETAILED_QUESTION = re.compile(
    r"为什么|怎么做|如何|解释|详细|比较|区别|步骤|原因|举例|"
    r"\b(?:why|how|explain|compare|difference|steps|detail)\b", re.IGNORECASE,
)


def chat_token_budget(user_text: str) -> int:
    """A bounded answer allowance, not a target length or extra model call."""
    return 256 if len(user_text) > 80 or _DETAILED_QUESTION.search(user_text) else 128


def public_chat_text(text: str) -> str:
    """Remove explicit provider reasoning envelopes before history/UI/speech.

    Never read a provider's separate reasoning field. An orphan closing tag can
    terminate a provider-prefilled reasoning prefix; discard everything before
    it. Unclosed/mismatched/partial envelopes fail closed, without retrying the
    model or logging hidden text. Ordinary words such as 'I think' are untouched.
    This is a protocol boundary, not a semantic classifier for untagged prose.
    """
    text = text.strip()
    parts, stack, cursor = [], [], 0
    for match in _REASONING_TAG.finditer(text):
        if not stack:
            parts.append(text[cursor:match.start()])
        closing, tag = match.group(1), match.group(2).lower()
        if not closing:
            stack.append(tag)
        elif not stack:
            parts.clear()
        elif stack[-1] != tag:
            return INCOMPLETE_CHAT_REPLY
        else:
            stack.pop()
        cursor = match.end()
    if stack:
        return INCOMPLETE_CHAT_REPLY
    parts.append(text[cursor:])
    answer = "".join(parts).strip()
    if not answer or _REASONING_FRAGMENT.search(answer) or _PARTIAL_REASONING_TAG.search(answer):
        return INCOMPLETE_CHAT_REPLY
    return answer


_SENTENCE_MARKS = frozenset("。！？!?")
_CLAUSE_MARKS = frozenset("，,；;：:")
_CLOSERS = frozenset("\"'”’」』）)]}")
_ABBREVIATION = re.compile(
    r"(?<![A-Za-z0-9_])(?:(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|a\.m|p\.m|Ph\.D)\."
    r"|(?:[A-Za-z]\.){2,}|[A-Za-z]\.)",
    re.IGNORECASE,
)


def _boundary_ends(text: str, *, clauses: bool) -> tuple[int, ...]:
    """Find punctuation boundaries without splitting numbers or abbreviations."""
    protected = {index for match in _ABBREVIATION.finditer(text)
                 for index in range(match.start(), match.end()) if text[index] == "."}
    protected.update(match.end() - 1 for match in re.finditer(r"(?m)^\s*\d+\.", text))
    ends = []
    index = 0
    while index < len(text):
        char = text[index]
        boundary = char in _SENTENCE_MARKS or (clauses and char in _CLAUSE_MARKS)
        if char == ".":
            before = text[index - 1] if index else ""
            after = text[index + 1] if index + 1 < len(text) else ""
            # Embedded periods include decimals, domains and dotted identifiers.
            embedded = bool(before and after and (
                (before.isdigit() and after.isdigit()) or
                (before.isascii() and after.isascii() and before.isalnum() and after.isalnum())))
            ellipsis = before == "." or after == "."
            boundary = index not in protected and not embedded and not ellipsis
        elif char in ",:" and index and index + 1 < len(text):
            if text[index - 1].isdigit() and text[index + 1].isdigit():
                boundary = False
        if boundary:
            end = index + 1
            while end < len(text) and (text[end] in _SENTENCE_MARKS or text[end] in _CLOSERS):
                end += 1
            ends.append(end)
            index = end
        else:
            index += 1
    return tuple(ends)


def finish_chat_text(text: str, finish_reason: str | None) -> str:
    """Keep normal replies intact; trim an explicitly token-limited last sentence.

    Call only for ordinary chat, never proposals or structured tool results.
    Missing punctuation alone is not evidence of truncation. No inference or
    additional model call is needed to handle a provider's ``length`` result.
    """
    text = public_chat_text(text)
    if finish_reason != "length":
        return text
    ends = _boundary_ends(text, clauses=False)
    return text[:ends[-1]].rstrip() if ends else INCOMPLETE_CHAT_REPLY


def limit_chat_text(text: str, max_chars: int) -> str:
    """Apply a separate spoken-answer budget without emitting half a sentence."""
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars <= 0:
        raise ValueError("chat text limit must be a positive integer")
    text = text.strip()
    if len(text) <= max_chars:
        return text
    ends = [end for end in _boundary_ends(text, clauses=False) if end <= max_chars]
    return text[:ends[-1]].rstrip() if ends else INCOMPLETE_CHAT_REPLY


def split_for_speech(text: str, first_limit: int = 40, next_limit: int = 64,
                     max_segments: int = 16) -> tuple[str, ...]:
    """Return bounded punctuation chunks, or ``()`` for the legacy path.

    Concatenating successful chunks exactly reconstructs ``text.strip()``.
    Prefer the first complete short sentence when another phrase follows.
    A short final phrase needs no punctuation. If any long phrase has no safe
    boundary, or the complete text would exceed the segment budget, no partial
    result is returned. English words, decimals and common abbreviations stay
    together; this helper never inserts punctuation or drops a text suffix.
    """
    if not isinstance(text, str):
        raise TypeError("speech text must be a string")
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
           for value in (first_limit, next_limit, max_segments)):
        raise ValueError("speech segment limits must be positive integers")
    text = text.strip()
    if not text or len(text) > first_limit + next_limit * (max_segments - 1):
        return ()
    boundaries = _boundary_ends(text, clauses=True)
    segments = []
    start = 0
    sentences = _boundary_ends(text, clauses=False)
    if max_segments > 1 and sentences:
        end = sentences[0]
        if end <= first_limit and any(char.isalnum() for char in text[end:]):
            while end < len(text) and text[end].isspace() and end < first_limit:
                end += 1
            segments.append(text[:end])
            start = end
    while start < len(text):
        if len(segments) >= max_segments:
            return ()
        limit = first_limit if not segments else next_limit
        if len(text) - start <= limit:
            segments.append(text[start:])
            break
        candidates = [end for end in boundaries if start < end <= start + limit]
        if not candidates:
            return ()
        end = candidates[-1]
        # Keep boundary whitespace in order when it fits in this chunk.
        while end < len(text) and text[end].isspace() and end - start < limit:
            end += 1
        segments.append(text[start:end])
        start = end
    return tuple(segments)
