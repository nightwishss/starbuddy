"""Operator-only candidate packaging CLI."""

from __future__ import annotations

import argparse,json
from pathlib import Path

from improvement_service import ImprovementClient
from .manager import PackageManager


def main():
    state=Path.home()/".local/state/vq-platform"
    parser=argparse.ArgumentParser(description="Package an evaluated candidate; does not deploy")
    parser.add_argument("--model-id",required=True); parser.add_argument("--candidate-version",required=True)
    parser.add_argument("--expected-sha256",required=True)
    parser.add_argument("--candidate-root",default=str(state/"candidates"))
    parser.add_argument("--package-root",default=str(state/"model-packages"))
    parser.add_argument("--key",default=str(state/"keys/model-package.key")); args=parser.parse_args()
    manager=PackageManager(args.candidate_root,args.package_root,args.key,ImprovementClient())
    print(json.dumps(manager.package(model_id=args.model_id,candidate_version=args.candidate_version,
                                     expected_sha256=args.expected_sha256),ensure_ascii=False,indent=2))
if __name__=="__main__": main()
