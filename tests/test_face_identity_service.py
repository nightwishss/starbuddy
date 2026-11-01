import unittest
from pathlib import Path

from face_identity_service.engine import CONSENT_VERSION, IdentityError, IdentityService, MODEL_ID


def candidate(pose="center", acceptable=True):
    return {"bbox":[.2,.2,.4,.5],"score":.95,"quality":.9,"sharpness":80,"pose":pose,
            "acceptable":acceptable,"rejection_reasons":[] if acceptable else ["image_too_blurry"],
            "embedding":[1.0]+[0.0]*127}


class FakeVault:
    def __init__(self): self.items=[]; self.templates=[]
    def create_identity(self, **values):
        item={"identity_id":"identity-test", "status":"active", "templates":[], **values}; self.items.append(item); return item
    def add_template(self, identity_id, model_id, embedding, **values): self.templates.append((identity_id,model_id,embedding,values))
    def identity(self, identity_id): return {**self.items[0],"templates":[{"template_id":f"t{i}"} for i,_ in enumerate(self.templates)]}
    def identities(self): return {"items":self.items}
    def delete_identity(self, identity_id, **values): self.items=[]; self.templates=[]; return {"deleted":True}
    def match(self, model_id, embedding): return {"status":"known","identity_id":"identity-test","profile_id":"owner","label":"April","similarity":.82}


class FakeEvents:
    def __init__(self): self.values=[]
    def publish(self,*args,**values): self.values.append((args,values))


class FakeOutput:
    def __init__(self): self.values=[]
    def say(self,*args,**values): self.values.append((args,values)); return {"job_id":"output_test"}


class FaceIdentityTests(unittest.TestCase):
    def service(self):
        vault,events,output=FakeVault(),FakeEvents(),FakeOutput()
        return IdentityService(object(),camera=object(),vault=vault,events=events,output=output),vault,events,output

    def test_enrollment_requires_explicit_current_consent(self):
        service,_,_,_=self.service()
        with self.assertRaises(IdentityError): service.start_enrollment(label="April",consent="user_confirmed",consent_version="old")

    def test_three_pose_collection_and_atomic_commit(self):
        service,vault,_,output=self.service()
        state=service.start_enrollment(label="April",profile_id="owner",consent="user_confirmed",consent_version=CONSENT_VERSION)
        session=service.enrollment
        service._collect(session,[candidate("center")],1)
        service._collect(session,[candidate("left")],2)
        service._collect(session,[candidate("right")],3)
        self.assertEqual(service._enrollment_public()["state"],"ready")
        self.assertNotIn("token", service.status()["enrollment"])
        result=service.commit_enrollment(state["token"],consent="user_confirmed")
        self.assertTrue(result["enrolled"]); self.assertFalse(result["raw_frames_retained"])
        self.assertEqual(len(vault.templates),3); self.assertTrue(all(item[1]==MODEL_ID for item in vault.templates))
        self.assertIsNone(service.enrollment)
        self.assertEqual([item[0][0] for item in output.values], ["好的，请正视摄像头", "很好，请稍向左转头", "很好，请稍向右转头", "采集完成，请点击确认保存", "April，你好，我已经记住你了"])

    def test_wrong_pose_and_multiple_faces_do_not_collect(self):
        service,_,_,_=self.service(); service.start_enrollment(label="A",consent="user_confirmed",consent_version=CONSENT_VERSION)
        service._collect(service.enrollment,[candidate("left")],1)
        service._collect(service.enrollment,[candidate(),candidate()],2)
        self.assertEqual(service._enrollment_public()["sample_count"],0)

    def test_cancel_cannot_race_a_commit(self):
        service,_,_,_=self.service(); state=service.start_enrollment(label="A",consent="user_confirmed",consent_version=CONSENT_VERSION)
        service.enrollment["state"]="committing"
        with self.assertRaises(IdentityError): service.cancel_enrollment(state["token"])
        self.assertIsNotNone(service.enrollment)

    def test_match_requires_three_hits_and_publishes_bounded_event(self):
        service,_,events,output=self.service()
        for _ in range(2): service._recognize([candidate()])
        self.assertEqual(service.status()["observation"]["status"],"confirming")
        service._recognize([candidate()]); result=service.status()["observation"]
        self.assertEqual(result["status"],"known"); self.assertEqual(result["label"],"April")
        self.assertNotIn("embedding",result); self.assertEqual(len(events.values),1)
        self.assertEqual(output.values, [])
        self.assertEqual(service.status()["recognition_response"], "context_only")

    def test_recognition_and_return_publish_context_without_speech(self):
        service,_,_,output=self.service()
        for _ in range(3): service._recognize([candidate()])
        service._recognize([]); service._recognize([]); service._recognize([])
        for _ in range(3): service._recognize([candidate()])
        self.assertEqual(output.values, [])
        self.assertEqual(service.status()["observation"]["status"], "known")

    def test_management_ui_can_cancel_enrollment_and_return_to_buddy(self):
        root = Path(__file__).parents[1] / "face_identity_service" / "static"
        html = (root / "index.html").read_text(encoding="utf-8")
        script = (root / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="return-buddy"', html)
        self.assertIn('sessionStorage.setItem("faceEnrollmentToken",token)', script)
        self.assertIn('post("/api/enrollment/cancel",{token})', script)
        self.assertIn('location.replace("http://127.0.0.1:8765/")', script)


if __name__ == "__main__": unittest.main()
