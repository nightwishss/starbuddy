"""Low-rate shadow inference; never changes the product posture decision."""

from __future__ import annotations

import hashlib
import math
import threading
import time

from training_service.logistic import probability


FEATURES=("head_forward_ratio","head_height_ratio","shoulder_slope_ratio","torso_lean_degrees")


class CanaryMonitor:
    def __init__(self,deployment,posture,*,interval=1.0,sleep=time.sleep):
        self.deployment,self.posture=deployment,posture; self.interval,self.sleep=interval,sleep
        self.lock=threading.Lock(); self.closed=False; self.thread=None; self.last_sequence=None
        self.cached_path=None; self.candidate=None
        self.value={"mode":"idle","observations":0,"selected":0,"compared":0,"agreements":0,
                    "agreement_rate":None,"last_prediction":None,"error":None,
                    "affects_product_decisions":False}

    def start(self):
        if self.thread and self.thread.is_alive(): return
        self.thread=threading.Thread(target=self._loop,name="vq-model-shadow-canary",daemon=True); self.thread.start()
    def close(self):
        self.closed=True
        if self.thread: self.thread.join(timeout=2)
    def status(self):
        with self.lock: return dict(self.value)
    def tick(self):
        state=self.deployment.status()
        if state.get("mode")!="shadow-canary":
            with self.lock: self.value.update({"mode":"idle","error":None})
            return self.status()
        package=self._candidate(state)
        posture=self.posture.status(); pose=posture.get("pose") or {}; sequence=pose.get("sequence")
        if not isinstance(sequence,int) or sequence==self.last_sequence: return self.status()
        self.last_sequence=sequence
        with self.lock: self.value["observations"]+=1; self.value["mode"]="shadow-canary"
        selected=int(hashlib.sha256(str(sequence).encode()).hexdigest()[:8],16)%100<state["canary_percent"]
        if not selected: return self.status()
        feature=(pose.get("posture_features") or {}).get("values") or {}
        try: row=[float(feature[name]) for name in FEATURES]
        except (KeyError,TypeError,ValueError): return self.status()
        if not all(math.isfinite(value) for value in row): return self.status()
        score=probability(package["classifier"],row); prediction="unhealthy" if score>=.5 else "healthy"
        baseline=posture.get("state"); compared=baseline in {"healthy","unhealthy"}
        with self.lock:
            self.value["selected"]+=1; self.value["last_prediction"]={"prediction":prediction,
                "unhealthy_probability":round(score,6),"baseline":baseline,"sequence":sequence}
            if compared:
                self.value["compared"]+=1; self.value["agreements"]+=int(prediction==baseline)
                self.value["agreement_rate"]=round(self.value["agreements"]/self.value["compared"],6)
            self.value["error"]=None
        return self.status()

    def _candidate(self,state):
        path=state["package_path"]
        if path!=self.cached_path:
            with open(path,"rb") as source: digest=hashlib.sha256(source.read()).hexdigest()
            if digest!=state.get("package_sha256"): raise ValueError("canary package digest mismatch")
            verified=self.deployment.packages.verify(path); payload=verified["payload"]
            if payload["model_id"]!=state["active_model"]:
                raise ValueError("canary package/state mismatch")
            self.candidate=payload["candidate"]; self.cached_path=path
            with self.lock:
                self.value.update({"observations":0,"selected":0,"compared":0,"agreements":0,
                                   "agreement_rate":None,"last_prediction":None,"error":None})
        return self.candidate

    def _loop(self):
        while not self.closed:
            try: self.tick()
            except Exception as exc:
                with self.lock: self.value.update({"error":type(exc).__name__,
                                                   "affects_product_decisions":False})
            self.sleep(self.interval)
