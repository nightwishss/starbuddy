"""Persistent shadow selector with Improvement-backed rollback."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path


class DeploymentError(ValueError):
    def __init__(self,code,message): super().__init__(message); self.code=code


class DeploymentManager:
    def __init__(self,packages,improvement,state_root,*,clock=time.time):
        self.packages,self.improvement=packages,improvement
        self.state_root=Path(state_root); self.clock=clock

    def start_shadow(self,*,package_path,baseline_model_id,canary_percent=10):
        if not isinstance(canary_percent,int) or not 1<=canary_percent<=25:
            raise DeploymentError("invalid_canary","canary must be 1..25 percent")
        package_path=Path(package_path).resolve()
        package_root=self.packages.package_root.resolve()
        if package_root not in package_path.parents:
            raise DeploymentError("package_location","package must be inside the managed package root")
        verified=self.packages.verify(package_path); payload=verified["payload"]
        task=payload["task"]; model_id=payload["model_id"]; run_id=payload["evaluation_run_id"]
        current=self.improvement.call("deployment.get",task=task)
        if current.get("mode")=="unmanaged":
            current=self.improvement.call("deployment.seed",task=task,model_id=baseline_model_id)
        elif current.get("mode")=="canary":
            raise DeploymentError("canary_active","roll back or complete the current canary first")
        if current.get("active_model")!=baseline_model_id:
            raise DeploymentError("baseline_mismatch","declared baseline is not active")
        package_digest=hashlib.sha256(package_path.read_bytes()).hexdigest()
        promoted=self.improvement.call("deployment.promote",task=task,model_id=model_id,
                                       run_id=run_id,canary_percent=canary_percent)
        state={"schema_version":1,"task":task,"mode":"shadow-canary","runtime_effect":"observe-only",
               "active_model":model_id,"baseline_model":baseline_model_id,"canary_percent":canary_percent,
               "package_path":str(package_path),"package_sha256":package_digest,
               "evaluation_run_id":run_id,"started_at":self.clock()}
        try: self._write(task,state)
        except Exception:
            try: self.improvement.call("deployment.rollback",task=task)
            except Exception: pass
            raise
        return {**state,"improvement":promoted,"affects_product_decisions":False}

    def rollback(self,*,task):
        result=self.improvement.call("deployment.rollback",task=task)
        state={"schema_version":1,"task":task,"mode":"rolled_back","runtime_effect":"none",
               "active_model":result["active_model"],"previous_model":result.get("previous_model"),
               "rolled_back_at":self.clock()}
        self._write(task,state)
        return {**state,"improvement":result,"affects_product_decisions":False}

    def status(self,task="posture.classifier"):
        path=self._path(task)
        if not path.exists(): return {"task":task,"mode":"unmanaged","runtime_effect":"none"}
        try: value=json.loads(path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError,UnicodeDecodeError) as exc:
            raise DeploymentError("invalid_state","deployment state is unreadable") from exc
        if not isinstance(value,dict) or value.get("task")!=task:
            raise DeploymentError("invalid_state","deployment state mismatch")
        return value

    def _path(self,task):
        if not isinstance(task,str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}",task):
            raise DeploymentError("invalid_task","invalid deployment task")
        return self.state_root/(task.replace(".","-")+".json")
    def _write(self,task,value):
        self.state_root.mkdir(parents=True,exist_ok=True); os.chmod(self.state_root,0o700)
        path=self._path(task); temporary=path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
        os.chmod(temporary,0o600); temporary.replace(path); return path
