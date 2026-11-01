"""Explicit feedback composition without raw media or transcript access."""

from __future__ import annotations

import math


class FeedbackError(ValueError):
    def __init__(self, code, message): super().__init__(message); self.code = code


POSTURE_LABELS={"healthy","unhealthy","unknown"}
VISION_VERDICTS={"correct","incorrect","partial"}


class FeedbackCollector:
    def __init__(self,features,improvement,posture):
        self.features,self.improvement,self.posture=features,improvement,posture

    @staticmethod
    def _label(value,allowed,field):
        if not isinstance(value,str) or value not in allowed:
            raise FeedbackError("invalid_label",f"invalid {field}")
        return value

    def label_current_posture(self,*,owner,label,trace_id=None,retention_days=30):
        label=self._label(label,POSTURE_LABELS,"posture label")
        status=self.posture.status(); pose=status.get("pose") or {}
        feature=pose.get("posture_features") or {}; values=feature.get("values") or {}
        age=pose.get("last_frame_age_ms")
        if not status.get("enabled") or not pose.get("ready") or not isinstance(age,(int,float)) or age>1000:
            raise FeedbackError("posture_not_ready","current posture sample is not fresh")
        names=("head_forward_ratio","head_height_ratio","shoulder_slope_ratio","torso_lean_degrees")
        if any(name not in values for name in names):
            raise FeedbackError("posture_not_ready","current posture features are incomplete")
        confidence=feature.get("confidence")
        clean={name:values[name] for name in names}; clean["confidence"]=confidence
        if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
               for value in clean.values()):
            raise FeedbackError("posture_not_ready","current posture features are invalid")
        record=self.features.write(owner=owner,schema_name="posture.features.v1",values=clean,
                                   source="posture-service",trace_id=trace_id,
                                   retention_days=retention_days)
        labelled=self._label_or_delete(owner,record,label)
        return {"feature":record,"label":labelled,"sample_age_ms":age,"raw_media_stored":False}

    def label_vision_summary(self,*,owner,counts,human_present,confidence,verdict,
                             trace_id=None,retention_days=30):
        verdict=self._label(verdict,VISION_VERDICTS,"vision verdict")
        record=self.features.write(owner=owner,schema_name="vision.objects.v1",
                                   values={"counts":counts,"human_present":human_present,
                                           "confidence":confidence},
                                   source="object-detector",trace_id=trace_id,
                                   retention_days=retention_days)
        labelled=self._label_or_delete(owner,record,verdict)
        return {"feature":record,"label":labelled,"raw_media_stored":False}

    def rate_conversation(self,*,model_id,rating,tags=None,trace_id=None):
        # No message text or note is accepted by this adapter.
        result=self.improvement.call("feedback.add",task="conversation.response",model_id=model_id,
                                     rating=rating,tags=tags or [],trace_id=trace_id)
        return {**result,"task":"conversation.response","raw_media_stored":False}

    def _label_or_delete(self,owner,record,label):
        try:
            return self.features.label(owner,record["feature_id"],label,label_source="explicit-user")
        except Exception:
            # A revoked consent or bad label must not leave an accidental unlabelled sample.
            try: self.features.delete(owner,record["feature_id"])
            except Exception: pass
            raise

    def stats(self):
        return {"feature_store":self.features.stats(),"improvement":self.improvement.stats(),
                "automatic_capture":False,"accepts_raw_media":False,"accepts_transcripts":False}
