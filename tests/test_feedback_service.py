import unittest

from feedback_service.collector import FeedbackCollector,FeedbackError


class Features:
    def __init__(self): self.writes=[]; self.labels=[]; self.deletes=[]; self.label_error=None
    def write(self,**values): self.writes.append(values); return {"feature_id":"feature_"+"a"*32}
    def label(self,*values,**options):
        self.labels.append((values,options))
        if self.label_error: raise self.label_error
        return {"label":values[2]}
    def delete(self,*values): self.deletes.append(values); return {"deleted":True}
    def stats(self): return {"features":len(self.writes)}

class Improvement:
    def __init__(self): self.calls=[]
    def call(self,operation,**values): self.calls.append((operation,values)); return {"feedback_id":"feedback_1"}
    def stats(self): return {"feedback":len(self.calls)}

class Posture:
    def __init__(self,age=20): self.age=age
    def status(self):
        return {"enabled":True,"pose":{"ready":True,"last_frame_age_ms":self.age,
                "posture_features":{"values":{"head_forward_ratio":.2,"head_height_ratio":.6,
                "shoulder_slope_ratio":.02,"shoulder_width":.3,"torso_lean_degrees":5.0},
                "confidence":.95}}}


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.features=Features(); self.improvement=Improvement()
        self.collector=FeedbackCollector(self.features,self.improvement,Posture())
    def test_posture_captures_fresh_schema_fields_only(self):
        result=self.collector.label_current_posture(owner="application:test",label="healthy")
        values=self.features.writes[0]["values"]
        self.assertNotIn("shoulder_width",values); self.assertEqual(values["confidence"],.95)
        self.assertFalse(result["raw_media_stored"])
    def test_posture_rejects_stale_sample_and_unknown_label(self):
        with self.assertRaises(FeedbackError):
            FeedbackCollector(self.features,self.improvement,Posture(age=1001)).label_current_posture(
                owner="application:test",label="healthy")
        with self.assertRaises(FeedbackError):
            self.collector.label_current_posture(owner="application:test",label="maybe")
        self.assertEqual(self.features.writes,[])
    def test_vision_accepts_only_summary_and_explicit_verdict(self):
        value=self.collector.label_vision_summary(owner="application:test",counts={"person":1},
                                                  human_present=True,confidence=.9,verdict="correct")
        self.assertEqual(self.features.writes[0]["schema_name"],"vision.objects.v1")
        self.assertFalse(value["raw_media_stored"])
    def test_label_failure_removes_partial_feature(self):
        self.features.label_error=RuntimeError("revoked")
        with self.assertRaises(RuntimeError):
            self.collector.label_vision_summary(owner="application:test",counts={},human_present=False,
                                                confidence=.8,verdict="correct")
        self.assertEqual(self.features.deletes[0][0],"application:test")
    def test_conversation_forwards_only_semantic_feedback(self):
        result=self.collector.rate_conversation(model_id="qwen3-4b",rating=1,
                                                tags=["natural"],trace_id="trace:1")
        operation,values=self.improvement.calls[0]
        self.assertEqual(operation,"feedback.add"); self.assertNotIn("note",values)
        self.assertEqual(values["task"],"conversation.response")
        self.assertFalse(result["raw_media_stored"])
    def test_stats_declares_non_automatic_boundary(self):
        value=self.collector.stats(); self.assertFalse(value["automatic_capture"])
        self.assertFalse(value["accepts_raw_media"]); self.assertFalse(value["accepts_transcripts"])


if __name__=="__main__": unittest.main()
