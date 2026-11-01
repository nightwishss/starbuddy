import hashlib,tempfile,unittest
from pathlib import Path

from training_service.executor import TrainingError,TrainingExecutor


class Features:
    def __init__(self,records): self.records=records
    def get_snapshot(self,owner,snapshot_id):
        import json
        canonical=json.dumps(sorted(self.records,key=lambda item:item["feature_id"]),ensure_ascii=False,
                             sort_keys=True,separators=(",", ":")).encode()
        return {"schema_name":"posture.features.v1","dataset_id":"posture-set-v1",
                "content_sha256":hashlib.sha256(canonical).hexdigest(),"records":self.records}

class Improvement:
    def __init__(self): self.calls=[]
    def call(self,operation,**values):
        self.calls.append((operation,values)); return {"run_id":"eval_1","passed":values["metrics"]["accuracy"]>=.75}

def records(count=12):
    result=[]
    for label,offset in (("healthy",0.0),("unhealthy",1.0)):
        for index in range(count):
            result.append({"feature_id":f"feature_{label}_{index:02d}","label":label,"values":{
                "head_forward_ratio":offset+index*.001,"head_height_ratio":1-offset*.2,
                "shoulder_slope_ratio":offset*.3,"torso_lean_degrees":offset*30}})
    return result


class TrainingTests(unittest.TestCase):
    def setUp(self): self.tmp=tempfile.TemporaryDirectory(); self.improvement=Improvement()
    def tearDown(self): self.tmp.cleanup()
    def test_trains_evaluates_and_writes_non_deploying_candidate(self):
        executor=TrainingExecutor(Features(records()),self.improvement,Path(self.tmp.name)/"candidates",clock=lambda:123)
        value=executor.train_posture(owner="application:test",snapshot_id="snapshot:1",
                                     model_id="posture-linear-personal",candidate_version="v1")
        self.assertTrue(value["evaluation"]["passed"]); self.assertFalse(value["deployed"])
        path=Path(value["artifact_path"]); self.assertEqual(path.stat().st_mode&0o777,0o600)
        self.assertEqual(self.improvement.calls[0][0],"evaluation.record")
    def test_requires_five_samples_per_class(self):
        executor=TrainingExecutor(Features(records(4)),self.improvement,Path(self.tmp.name)/"candidates")
        with self.assertRaises(TrainingError) as caught:
            executor.train_posture(owner="application:test",snapshot_id="snapshot:1",
                                   model_id="posture-linear-personal",candidate_version="v1")
        self.assertEqual(caught.exception.code,"insufficient_dataset")
    def test_never_overwrites_candidate_version(self):
        executor=TrainingExecutor(Features(records()),self.improvement,Path(self.tmp.name)/"candidates")
        values=dict(owner="application:test",snapshot_id="snapshot:1",
                    model_id="posture-linear-personal",candidate_version="v1")
        executor.train_posture(**values)
        with self.assertRaises(TrainingError) as caught: executor.train_posture(**values)
        self.assertEqual(caught.exception.code,"candidate_exists")
    def test_rejects_snapshot_digest_mismatch(self):
        source=Features(records()); original=source.get_snapshot
        source.get_snapshot=lambda *args: {**original(*args),"content_sha256":"0"*64}
        executor=TrainingExecutor(source,self.improvement,Path(self.tmp.name)/"candidates")
        with self.assertRaises(TrainingError) as caught:
            executor.train_posture(owner="application:test",snapshot_id="snapshot:1",
                                   model_id="posture-linear-personal",candidate_version="v1")
        self.assertEqual(caught.exception.code,"dataset_integrity")


if __name__=="__main__": unittest.main()
