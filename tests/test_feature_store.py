import tempfile
import unittest
from pathlib import Path

from feature_store_service.store import FeatureError,FeatureStore


POSTURE={"head_forward_ratio":.2,"head_height_ratio":.6,"shoulder_slope_ratio":.02,
         "torso_lean_degrees":5.0,"confidence":.95}


class FeatureStoreTests(unittest.TestCase):
    def setUp(self):
        self.allowed=set(); self.tmp=tempfile.TemporaryDirectory()
        self.store=FeatureStore(Path(self.tmp.name)/"features.sqlite3",
                                consent=lambda scope: scope in self.allowed,clock=lambda:1000)
    def tearDown(self): self.store.close(); self.tmp.cleanup()
    def test_write_requires_schema_specific_consent_and_rejects_unknown_fields(self):
        with self.assertRaises(FeatureError) as caught:
            self.store.write(owner="app:posture",schema_name="posture.features.v1",
                             values=POSTURE,source="pose")
        self.assertEqual(caught.exception.code,"consent_required")
        self.allowed.add("feature_store.posture.features.v1")
        value=self.store.write(owner="app:posture",schema_name="posture.features.v1",
                               values=POSTURE,source="pose")
        self.assertFalse(value["raw_media_stored"])
        with self.assertRaises(FeatureError):
            self.store.write(owner="app:posture",schema_name="posture.features.v1",
                             values={**POSTURE,"image":"base64"},source="pose")
    def test_owner_scoped_label_query_delete(self):
        self.allowed.add("feature_store.posture.features.v1")
        feature=self.store.write(owner="app:posture",schema_name="posture.features.v1",
                                 values=POSTURE,source="pose")
        self.store.label(owner="app:posture",feature_id=feature["feature_id"],label="healthy")
        values=self.store.query(owner="app:posture",schema_name="posture.features.v1",labelled=True)
        self.assertEqual(values["features"][0]["label"],"healthy")
        self.assertEqual(self.store.query(owner="other",schema_name="posture.features.v1")["features"],[])
        self.assertTrue(self.store.delete(owner="app:posture",feature_id=feature["feature_id"])["deleted"])
        self.assertEqual(self.store.stats()["labelled"],0)
    def test_snapshot_is_canonical_metadata_not_media(self):
        self.allowed.add("feature_store.posture.features.v1")
        for label in ("healthy","unhealthy"):
            feature=self.store.write(owner="app:posture",schema_name="posture.features.v1",
                                     values=POSTURE,source="pose")
            self.store.label(owner="app:posture",feature_id=feature["feature_id"],label=label)
        result=self.store.snapshot(owner="app:posture",schema_name="posture.features.v1",
                                   dataset_id="posture-personal-v1")
        self.assertEqual(result["item_count"],2); self.assertEqual(len(result["content_sha256"]),64)
        self.assertEqual(result["task"],"posture.classifier")
        self.assertFalse(result["raw_media_stored"])
        first=self.store.query(owner="app:posture",schema_name="posture.features.v1")["features"][0]
        self.store.delete(owner="app:posture",feature_id=first["feature_id"])
        frozen=self.store.get_snapshot(owner="app:posture",snapshot_id=result["snapshot_id"])
        self.assertEqual(len(frozen["records"]),2)
        self.assertEqual(frozen["content_sha256"],result["content_sha256"])


if __name__=="__main__": unittest.main()
