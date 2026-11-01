"""Conservative use of fresh facial-expression cues in conversation."""

from __future__ import annotations

import dataclasses
import re
import time


POSITIVE = {"happy_like"}
NEGATIVE = {"sad_like", "angry_like", "disgust_like", "fear_like", "contempt_like"}
STATE_TERMS = ("心情", "情绪", "表情", "状态", "看起来", "感觉怎么样")
SELF_TERMS = ("我", "咱")
QUESTION_TERMS = ("?", "？", "吗", "呢", "如何", "怎么样", "看看我", "你觉得我")


@dataclasses.dataclass(frozen=True)
class AffectDecision:
    direct_answer: str | None = None
    tone_hint: str | None = None
    visual_style: str = "speaking"
    cue_label: str | None = None


class AffectInteractionPolicy:
    """Never diagnoses mood or turns a model label into an autonomous action."""

    def __init__(self, *, clock=time.time, min_confidence=.75):
        self.clock = clock
        self.min_confidence = min_confidence

    @staticmethod
    def asks_about_self(text):
        return (any(term in text for term in SELF_TERMS)
                and any(term in text for term in STATE_TERMS)
                and any(term in text for term in QUESTION_TERMS))

    @staticmethod
    def answer_is_bounded(answer) -> bool:
        """Conservative wording guard, not a complete semantic safety proof.

        Require a visual-cue boundary and uncertainty/self-report precedence;
        obvious mental-state assertions fail even beside a disclaimer.
        """
        if not isinstance(answer, str) or not answer.strip():
            return False
        text = " ".join(answer.casefold().replace("’", "'").split())
        hard_zh = (r"(?:你|您)(?:现在|目前|此刻|其实|内心|真的|确实|就是|一定|肯定|绝对|"
                   r"非常|很|正|感到|是|的心情|的情绪)*"
                   r"(?:开心|高兴|快乐|难过|伤心|悲伤|愤怒|生气|焦虑|抑郁|痛苦|害怕|"
                   r"恐惧|孤独|沮丧|平静|放松|紧张|轻松|失落|烦躁|幸福)")
        hard_en = (r"\byou(?:\s+are|'re|\s+feel)(?:\s+(?:now|currently|really|very|"
                   r"definitely|certainly|clearly|so|truly|not))*\s+"
                   r"(?:sad|happy|angry|depressed|anxious|afraid|scared|upset|lonely|"
                   r"calm|relaxed|stressed|miserable|unhappy|nervous|joyful)\b")
        if re.search(hard_zh, text) or re.search(hard_en, text):
            return False
        cue = re.search(r"表情|视觉线索|画面线索|\b(?:expression|facial|visual cues?)\b", text)
        bounded = re.search(
            r"可能|似乎|不一定|未必|不能.{0,10}(?:代表|说明|确定|推断)|不等于|"
            r"不(?:会|能).{0,8}(?:否定|判断|断定).{0,8}(?:感受|心情)|"
            r"(?:感受|自述|说的).{0,12}为准|只能.{0,8}(?:线索|猜测)|"
            r"\b(?:may|might|seems?|suggests?|uncertain)\b|"
            r"(?:doesn't|does not|cannot|can't|can not).{0,30}(?:prove|reflect|tell|determine|represent)|"
            r"\b(?:self[- ]report|own feelings)\b", text)
        return bool(cue and bounded)

    def decide(self, text, snapshot):
        explicit = self.asks_about_self(text)
        cue = self._current_cue(snapshot)
        if explicit:
            if cue is None:
                return AffectDecision(
                    direct_answer="目前我没有可靠的表情线索。你可以告诉我自己感觉如何，我会认真听。",
                    visual_style="attentive",
                )
            label = cue["top_label"]
            if label == "neutral":
                return AffectDecision(
                    direct_answer="从镜头里看，表情线索比较中性；这不能说明真实心情。你自己感觉怎么样？",
                    visual_style="attentive", cue_label=label,
                )
            if label in POSITIVE:
                return AffectDecision(
                    direct_answer="从镜头里看，表情似乎比较轻松。不过我只能根据画面猜测，你自己感觉怎么样？",
                    visual_style="curious", cue_label=label,
                )
            if label in NEGATIVE:
                return AffectDecision(
                    direct_answer="从镜头里看，表情线索有点紧绷，但我不一定看得准。你自己感觉怎么样？",
                    visual_style="gentle", cue_label=label,
                )
            return AffectDecision(
                direct_answer="镜头里的表情似乎有些变化，不过我不一定看得准。你想说说现在的感觉吗？",
                visual_style="attentive", cue_label=label,
            )
        # Subtle presentation only. The cue is never named in an unsolicited answer.
        if cue and cue["top_label"] in NEGATIVE:
            return AffectDecision(
                tone_hint="这一轮回答语气温和、简短，少追问；不要提到摄像头、表情或推测用户情绪。",
                cue_label=cue["top_label"],
            )
        return AffectDecision()

    def _current_cue(self, snapshot):
        if not isinstance(snapshot, dict):
            return None
        now = self.clock()
        cues = snapshot.get("affect_cues") or []
        if len(cues) != 1 or not isinstance(cues[0], dict):
            return None
        cue = cues[0]
        if cue.get("source") != "camera.front" or cue.get("confidence", 0) < self.min_confidence:
            return None
        if not isinstance(cue.get("expires_at"), (int, float)) or cue["expires_at"] <= now:
            return None
        people = [person for person in snapshot.get("people", [])
                  if person.get("source") == "camera.front"]
        if len(people) != 1 or people[0].get("track_id") != cue.get("track_id"):
            return None
        return cue
