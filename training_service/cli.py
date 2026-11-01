"""Operator CLI for bounded offline training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_store_service import FeatureStoreClient
from improvement_service import ImprovementClient

from .executor import TrainingExecutor


def main():
    parser=argparse.ArgumentParser(description="Train an evaluated lightweight candidate; never deploy it")
    parser.add_argument("posture",choices=["posture"]); parser.add_argument("--owner",required=True)
    parser.add_argument("--snapshot-id",required=True); parser.add_argument("--model-id",required=True)
    parser.add_argument("--candidate-version",required=True)
    parser.add_argument("--artifact-root",default=str(Path.home()/".local/state/vq-platform/candidates"))
    args=parser.parse_args()
    executor=TrainingExecutor(FeatureStoreClient(),ImprovementClient(),args.artifact_root)
    print(json.dumps(executor.train_posture(owner=args.owner,snapshot_id=args.snapshot_id,
          model_id=args.model_id,candidate_version=args.candidate_version),ensure_ascii=False,indent=2))


if __name__=="__main__": main()
