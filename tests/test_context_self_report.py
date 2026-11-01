"""Self-reported feelings stay dialogue; only explicit live cue questions route."""

import json
import unittest

from interaction_context.query import perception_query
from interaction_context.reasoning import build_reasoning_packet


class ContextSelfReportTests(unittest.TestCase):
    def test_current_self_reports_remain_normal_dialogue(self):
        for text in (
            "我今天心情不好，陪我聊聊", "我的情绪很平静", "我心情不错",
            "我今天心情不好，可以陪我聊聊吗？", "我今天心情不好，如何缓解？",
            "我看起来有点累，但其实很开心", "我的表情很严肃，实际上我没生气",
            "My mood is great today", "My expression is neutral.",
            "My mood is low, can you talk with me?", "How can I improve my mood?",
        ):
            with self.subTest(text=text):
                self.assertEqual(perception_query(text), "chat")

    def test_history_and_past_questions_do_not_request_current_camera_cues(self):
        for text in (
            "我昨天心情不好", "我刚才的情绪有点低落", "昨天我的表情怎么样？",
            "你之前问我现在心情如何", "我以前表情总是比较严肃",
            "昨天，你问我现在心情如何", "昨天，我的表情怎么样？",
            "My mood was low yesterday", "What was my mood yesterday?",
            "Yesterday you asked how do I look", "How is my mood compared with last week?",
            "Yesterday, you asked how do I look?",
        ):
            with self.subTest(text=text):
                self.assertEqual(perception_query(text), "chat")

    def test_explicit_current_questions_still_route_affect(self):
        for text in (
            "我今天心情如何", "你觉得我现在状态怎么样？", "你看我现在表情怎么样？",
            "你说我现在表情怎么样？",
            "看看我的表情", "判断一下我现在的表情", "我的表情自然吗？",
            "我现在看起来怎么样", "What is my mood right now?", "How is my mood?",
            "我现在是什么表情？", "我现在是什么心情？", "How am I looking?",
            "How do I look?", "What does my expression look like?",
            "Can you describe my expression?", "Look at my facial expression.",
            "Can you tell me what my mood is?",
            "我昨天心情不好，现在看看我的表情", "I was tired yesterday. How do I look now?",
        ):
            with self.subTest(text=text):
                self.assertEqual(perception_query(text), "affect")

    def test_existing_non_affect_routes_are_unchanged(self):
        for text, expected in (
            ("你能看到我吗", "presence"), ("Can you see my face?", "presence"),
            ("你看到什么", "scene"), ("房间里有几个人", "scene"),
            ("我现在坐姿怎么样", "posture"), ("Check my posture", "posture"),
            ("我是谁", "identity"), ("Who am I?", "identity"),
            ("你知道我在哪里吗", "location"), ("看看我的代码", "chat"),
        ):
            with self.subTest(text=text):
                self.assertEqual(perception_query(text), expected)

    def test_self_report_packet_does_not_expose_visual_affect_label(self):
        meta = {"source": "camera.front", "observed_at": 1000,
                "expires_at": 1003, "confidence": .9}
        context = {
            "captured_at": 1000,
            "perception": {"people": [{**meta, "track_id": "synthetic"}],
                           "affect_cues": [{**meta, "track_id": "synthetic", "top_label": "sad_like"}]},
            "evidence": {"people": {"state": "available"}, "affect": {"state": "available"}},
        }
        for text in ("我今天心情很好", "My mood is great today", "我昨天心情不好"):
            with self.subTest(text=text):
                packet = build_reasoning_packet({"user_text": text}, context, now=1001)
                self.assertEqual(packet["user_input"], text)
                self.assertNotIn("sad_like", json.dumps(packet))
                self.assertIn("tone_hint", packet["observations"]["affect"])
        packet = build_reasoning_packet({"user_text": "我现在表情怎么样？"}, context, now=1001)
        self.assertEqual(packet["observations"]["affect"]["top_label"], "sad_like")


if __name__ == "__main__":
    unittest.main()
