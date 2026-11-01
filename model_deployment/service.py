"""Owner-only status endpoint for the shadow canary monitor."""

from __future__ import annotations

import argparse,asyncio,json,os
from pathlib import Path

from improvement_service import ImprovementClient
from model_package import PackageManager
from posture_service import PostureClient
from .manager import DeploymentManager
from .monitor import CanaryMonitor


DEFAULT_SOCKET=os.environ.get("VQ_CANARY_SOCKET",f"/run/user/{os.getuid()}/vq-model-canary.sock")


async def run(socket_path):
    os.umask(0o077); state=Path.home()/".local/state/vq-platform"
    improvement=ImprovementClient(); packages=PackageManager(state/"candidates",state/"model-packages",
                                                              state/"keys/model-package.key",improvement)
    deployment=DeploymentManager(packages,improvement,state/"deployments")
    monitor=CanaryMonitor(deployment,PostureClient()); monitor.start()
    path=Path(socket_path); path.unlink(missing_ok=True)
    async def handle(reader,writer):
        try:
            raw=await reader.readline()
            if not raw or len(raw)>4096 or not raw.endswith(b"\n"): raise ValueError("invalid framing")
            request=json.loads(raw)
            if request!={"operation":"status"}: raise ValueError("status only")
            value={"ok":True,"result":monitor.status()}
        except Exception as exc: value={"ok":False,"code":"invalid_request","error":str(exc)}
        writer.write((json.dumps(value,separators=(",", ":"))+"\n").encode()); await writer.drain()
        writer.close(); await writer.wait_closed()
    server=await asyncio.start_unix_server(handle,path=socket_path); os.chmod(path,0o600)
    try:
        async with server: await server.serve_forever()
    finally: monitor.close(); path.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--socket",default=DEFAULT_SOCKET)
    args=parser.parse_args(); asyncio.run(run(args.socket))
if __name__=="__main__": main()
