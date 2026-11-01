"""Operator CLI for shadow canary selection and rollback."""

from __future__ import annotations

import argparse,json
from pathlib import Path

from improvement_service import ImprovementClient
from model_package import PackageManager
from .manager import DeploymentManager


def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="operation",required=True)
    start=sub.add_parser("start-shadow"); start.add_argument("--package",required=True)
    start.add_argument("--baseline-model-id",required=True); start.add_argument("--percent",type=int,default=10)
    rollback=sub.add_parser("rollback"); rollback.add_argument("--task",default="posture.classifier")
    status=sub.add_parser("status"); status.add_argument("--task",default="posture.classifier")
    args=parser.parse_args(); root=Path.home()/".local/state/vq-platform"; improvement=ImprovementClient()
    packages=PackageManager(root/"candidates",root/"model-packages",root/"keys/model-package.key",improvement)
    manager=DeploymentManager(packages,improvement,root/"deployments")
    if args.operation=="start-shadow":
        value=manager.start_shadow(package_path=args.package,baseline_model_id=args.baseline_model_id,
                                   canary_percent=args.percent)
    elif args.operation=="rollback": value=manager.rollback(task=args.task)
    else: value=manager.status(args.task)
    print(json.dumps(value,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
