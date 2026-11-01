import hashlib,json,tempfile,unittest
from pathlib import Path

from model_package import PackageError,PackageManager


class Improvement:
    def call(self,operation,**values):
        return {"task":"posture.classifier","model_id":"posture-linear-personal",
                "candidate_version":"v1","passed":True,"run_id":values["run_id"]}


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.candidates=self.root/"candidates"; path=self.candidates/"posture-linear-personal"/"v1.json"
        path.parent.mkdir(parents=True)
        self.candidate={"format":"vq-linear-classifier-v1","task":"posture.classifier",
                        "model_id":"posture-linear-personal","candidate_version":"v1",
                        "deployable":False,"classifier":{"weights":[1]},
                        "evaluation":{"run_id":"eval_1","passed":True}}
        path.write_text(json.dumps(self.candidate,ensure_ascii=False,sort_keys=True,indent=2)+"\n")
        self.path=path; self.digest=hashlib.sha256(path.read_bytes()).hexdigest()
        self.manager=PackageManager(self.candidates,self.root/"packages",self.root/"keys/key",Improvement())
    def tearDown(self): self.tmp.cleanup()
    def test_packages_and_verifies_exact_candidate(self):
        result=self.manager.package(model_id="posture-linear-personal",candidate_version="v1",
                                    expected_sha256=self.digest)
        package=Path(result["package_path"]); self.assertEqual(package.stat().st_mode&0o777,0o600)
        self.assertTrue(self.manager.verify(package)["verified"]); self.assertFalse(result["runtime_applied"])
    def test_rejects_candidate_digest_mismatch(self):
        with self.assertRaises(PackageError) as caught:
            self.manager.package(model_id="posture-linear-personal",candidate_version="v1",
                                 expected_sha256="0"*64)
        self.assertEqual(caught.exception.code,"candidate_integrity")
    def test_detects_package_tampering(self):
        result=self.manager.package(model_id="posture-linear-personal",candidate_version="v1",
                                    expected_sha256=self.digest)
        package=Path(result["package_path"]); value=json.loads(package.read_text())
        value["payload"]["candidate"]["classifier"]["weights"]=[9]
        package.write_text(json.dumps(value))
        with self.assertRaises(PackageError) as caught: self.manager.verify(package)
        self.assertEqual(caught.exception.code,"invalid_signature")


if __name__=="__main__": unittest.main()
