"""Snapshot-to-candidate executor; never deploys or loads the candidate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

from .logistic import fit,metrics


IDENTIFIER=re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
FEATURES=("head_forward_ratio","head_height_ratio","shoulder_slope_ratio","torso_lean_degrees")
LABELS={"healthy":0,"unhealthy":1}


class TrainingError(ValueError):
    def __init__(self,code,message): super().__init__(message); self.code=code


class TrainingExecutor:
    def __init__(self,features,improvement,artifact_root,*,clock=time.time):
        self.features,self.improvement=features,improvement
        self.artifact_root=Path(artifact_root); self.clock=clock

    @staticmethod
    def _id(value,field):
        if not isinstance(value,str) or not IDENTIFIER.fullmatch(value):
            raise TrainingError("invalid_identifier",f"invalid {field}")

    @staticmethod
    def _partition(records):
        by_label={0:[],1:[]}
        for record in sorted(records,key=lambda item:item["feature_id"]):
            if record.get("label") not in LABELS: continue
            values=record.get("values") or {}
            try: row=[float(values[name]) for name in FEATURES]
            except (KeyError,TypeError,ValueError) as exc:
                raise TrainingError("invalid_dataset","missing numeric posture feature") from exc
            if not all(math.isfinite(value) for value in row):
                raise TrainingError("invalid_dataset","non-finite posture feature")
            by_label[LABELS[record["label"]]].append((record["feature_id"],row))
        if any(len(items)<5 for items in by_label.values()):
            raise TrainingError("insufficient_dataset","at least five samples per class required")
        train=[]; test=[]
        for label,items in by_label.items():
            test_count=max(1,round(len(items)*.2))
            ranked=sorted(items,key=lambda item:hashlib.sha256(item[0].encode()).hexdigest())
            test.extend((row,label) for _,row in ranked[:test_count])
            train.extend((row,label) for _,row in ranked[test_count:])
        return train,test

    def train_posture(self,*,owner,snapshot_id,model_id,candidate_version,
                      thresholds=None):
        for value,field in ((owner,"owner"),(snapshot_id,"snapshot_id"),(model_id,"model_id"),
                            (candidate_version,"candidate_version")): self._id(value,field)
        candidate_path=self.artifact_root/model_id/f"{candidate_version}.json"
        if candidate_path.exists(): raise TrainingError("candidate_exists","candidate version already exists")
        snapshot=self.features.get_snapshot(owner,snapshot_id)
        if snapshot.get("schema_name")!="posture.features.v1":
            raise TrainingError("unsupported_schema","posture.features.v1 snapshot required")
        records=snapshot.get("records") or []
        canonical=json.dumps(sorted(records,key=lambda item:item["feature_id"]),ensure_ascii=False,
                             sort_keys=True,separators=(",", ":")).encode()
        if hashlib.sha256(canonical).hexdigest()!=snapshot.get("content_sha256"):
            raise TrainingError("dataset_integrity","snapshot content digest mismatch")
        train,test=self._partition(records)
        trained=fit([row for row,_ in train],[label for _,label in train])
        measured=metrics(trained,[row for row,_ in test],[label for _,label in test])
        thresholds=thresholds or {"accuracy":.75,"balanced_accuracy":.75}
        artifact={"format":"vq-linear-classifier-v1","task":"posture.classifier",
                  "model_id":model_id,"candidate_version":candidate_version,
                  "input_schema":"posture.features.v1","feature_order":list(FEATURES),
                  "label_mapping":{"healthy":0,"unhealthy":1},"classifier":trained,
                  "dataset":{"dataset_id":snapshot["dataset_id"],"snapshot_id":snapshot_id,
                             "content_sha256":snapshot["content_sha256"]},
                  "split":{"strategy":"sha256-stratified-80-20","train_samples":len(train),
                           "test_samples":len(test)},"metrics":measured,
                  "created_at":self.clock(),"deployable":False}
        evaluation=self.improvement.call("evaluation.record",task="posture.classifier",
                                         model_id=model_id,candidate_version=candidate_version,
                                         dataset_id=snapshot["dataset_id"],metrics=measured,
                                         thresholds=thresholds)
        artifact["evaluation"]=evaluation
        path,digest=self._write(candidate_path,artifact)
        return {"artifact_path":str(path),"artifact_sha256":digest,
                "metrics":measured,"evaluation":evaluation,"deployed":False,
                "automatic_promotion":False}

    def _write(self,path,artifact):
        directory=path.parent; directory.mkdir(parents=True,exist_ok=True)
        os.chmod(self.artifact_root,0o700); os.chmod(directory,0o700)
        body=(json.dumps(artifact,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()
        try: descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        except FileExistsError as exc: raise TrainingError("candidate_exists","candidate version already exists") from exc
        try:
            with os.fdopen(descriptor,"wb") as output: output.write(body)
        except Exception:
            path.unlink(missing_ok=True); raise
        return path,hashlib.sha256(body).hexdigest()
