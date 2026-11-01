import hashlib, tempfile, unittest
from pathlib import Path
from improvement_service.store import ImprovementError, ImprovementStore

class ImprovementTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.store=ImprovementStore(Path(self.tmp.name)/"db.sqlite3",clock=lambda:123)
    def tearDown(self): self.store.close(); self.tmp.cleanup()
    def test_feedback_requires_consent_and_stores_no_media(self):
        with self.assertRaises(ImprovementError) as caught:
            self.store.add_feedback(task="vision.pose",model_id="pose-v1",rating=1)
        self.assertEqual(caught.exception.code,"consent_required")
        self.store.set_consent(scope="improvement.feedback",allowed=True)
        value=self.store.add_feedback(task="vision.pose",model_id="pose-v1",rating=1,tags=["accurate"])
        self.assertFalse(value["raw_media_stored"])
    def test_promotion_requires_dataset_and_passing_evaluation_then_rolls_back(self):
        self.store.set_consent(scope="improvement.datasets",allowed=True)
        digest=hashlib.sha256(b"semantic-fixture-v1").hexdigest()
        self.store.register_dataset(dataset_id="pose-set-v1",task="vision.pose",item_count=20,
                                    schema_name="pose.metrics.v1",content_sha256=digest)
        failed=self.store.record_evaluation(task="vision.pose",model_id="pose-v2",candidate_version="v2",
                                            dataset_id="pose-set-v1",metrics={"accuracy":.7},thresholds={"accuracy":.8})
        with self.assertRaises(ImprovementError):
            self.store.promote(task="vision.pose",model_id="pose-v2",run_id=failed["run_id"])
        passed=self.store.record_evaluation(task="vision.pose",model_id="pose-v1",candidate_version="v1",
                                            dataset_id="pose-set-v1",metrics={"accuracy":.9},thresholds={"accuracy":.8})
        self.assertTrue(self.store.evaluation(run_id=passed["run_id"])["passed"])
        self.store.seed_deployment(task="vision.pose",model_id="pose-production")
        self.store.promote(task="vision.pose",model_id="pose-v1",run_id=passed["run_id"])
        passed2=self.store.record_evaluation(task="vision.pose",model_id="pose-v2",candidate_version="v2",
                                             dataset_id="pose-set-v1",metrics={"accuracy":.92},thresholds={"accuracy":.8})
        self.store.promote(task="vision.pose",model_id="pose-v2",run_id=passed2["run_id"])
        self.assertEqual(self.store.rollback(task="vision.pose")["active_model"],"pose-v1")
    def test_seed_is_idempotent_and_rejects_different_baseline(self):
        first=self.store.seed_deployment(task="posture.classifier",model_id="posture-policy-v1")
        second=self.store.seed_deployment(task="posture.classifier",model_id="posture-policy-v1")
        self.assertEqual(first["active_model"],second["active_model"])
        with self.assertRaises(ImprovementError) as caught:
            self.store.seed_deployment(task="posture.classifier",model_id="other")
        self.assertEqual(caught.exception.code,"deployment_exists")
    def test_dataset_registration_is_idempotent_but_rejects_conflicts(self):
        self.store.set_consent(scope="improvement.datasets",allowed=True)
        digest=hashlib.sha256(b"semantic-fixture-v1").hexdigest()
        first=self.store.register_dataset(dataset_id="pose-set-v1",task="vision.pose",item_count=20,
                                          schema_name="pose.metrics.v1",content_sha256=digest)
        second=self.store.register_dataset(dataset_id="pose-set-v1",task="vision.pose",item_count=20,
                                           schema_name="pose.metrics.v1",content_sha256=digest)
        self.assertFalse(first["existing"]); self.assertTrue(second["existing"])
        with self.assertRaises(ImprovementError) as caught:
            self.store.register_dataset(dataset_id="pose-set-v1",task="vision.pose",item_count=21,
                                        schema_name="pose.metrics.v1",content_sha256=digest)
        self.assertEqual(caught.exception.code,"dataset_conflict")
    def test_stats_make_automation_boundary_explicit(self):
        status=self.store.stats(); self.assertFalse(status["automatic_training"]); self.assertFalse(status["automatic_promotion"])

if __name__ == "__main__": unittest.main()
