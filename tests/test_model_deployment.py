import json,tempfile,unittest
from pathlib import Path

from model_deployment import CanaryMonitor,DeploymentError,DeploymentManager


class Packages:
    def __init__(self,root): self.package_root=root; self.root=root; root.mkdir(parents=True); self.path=root/"m.vqmodel"; self.path.write_text("signed")
    def verify(self,path):
        return {"payload":{"task":"posture.classifier","model_id":"candidate-v1",
                "evaluation_run_id":"eval_1","candidate":{"classifier":{
                "means":[0,0,0,0],"scales":[1,1,1,1],"weights":[1,0,0,0],"bias":0}}}}

class Improvement:
    def __init__(self): self.state={"task":"posture.classifier","mode":"unmanaged","active_model":None}; self.calls=[]
    def call(self,operation,**values):
        self.calls.append((operation,values))
        if operation=="deployment.get": return dict(self.state)
        if operation=="deployment.seed": self.state={"task":values["task"],"mode":"production","active_model":values["model_id"]}; return dict(self.state)
        if operation=="deployment.promote":
            previous=self.state["active_model"]; self.state={"task":values["task"],"mode":"canary","active_model":values["model_id"],"previous_model":previous}; return dict(self.state)
        if operation=="deployment.rollback":
            self.state={"task":values["task"],"mode":"rolled_back","active_model":self.state["previous_model"],"previous_model":self.state["active_model"]}; return dict(self.state)

class Posture:
    def __init__(self): self.sequence=1
    def status(self):
        return {"state":"unhealthy","pose":{"sequence":self.sequence,"posture_features":{"values":{
                "head_forward_ratio":1,"head_height_ratio":0,"shoulder_slope_ratio":0,
                "torso_lean_degrees":0}}}}


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); root=Path(self.tmp.name); self.packages=Packages(root/"packages")
        self.improvement=Improvement(); self.manager=DeploymentManager(self.packages,self.improvement,root/"state",clock=lambda:123)
    def tearDown(self): self.tmp.cleanup()
    def test_shadow_canary_seeds_baseline_and_is_rollbackable(self):
        value=self.manager.start_shadow(package_path=self.packages.path,baseline_model_id="posture-policy-v1",canary_percent=10)
        self.assertFalse(value["affects_product_decisions"]); self.assertEqual(self.manager.status()["mode"],"shadow-canary")
        rolled=self.manager.rollback(task="posture.classifier")
        self.assertEqual(rolled["active_model"],"posture-policy-v1")
    def test_rejects_nested_canary(self):
        self.manager.start_shadow(package_path=self.packages.path,baseline_model_id="posture-policy-v1")
        with self.assertRaises(DeploymentError) as caught:
            self.manager.start_shadow(package_path=self.packages.path,baseline_model_id="candidate-v1")
        self.assertEqual(caught.exception.code,"canary_active")
    def test_monitor_runs_selected_candidate_without_changing_decision(self):
        self.manager.start_shadow(package_path=self.packages.path,baseline_model_id="posture-policy-v1",canary_percent=25)
        posture=Posture(); monitor=CanaryMonitor(self.manager,posture)
        for sequence in range(1,100): posture.sequence=sequence; monitor.tick()
        value=monitor.status(); self.assertGreater(value["selected"],0); self.assertGreater(value["compared"],0)
        self.assertFalse(value["affects_product_decisions"])


if __name__=="__main__": unittest.main()
