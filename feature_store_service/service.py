"""Owner-only Feature Store API with consent delegated to Improvement policy."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from improvement_service import ImprovementClient

from .client import DEFAULT_SOCKET
from .store import FeatureError, FeatureStore


DEFAULT_DB=Path.home()/".local/state/vq-platform/features.sqlite3"


class Service:
    def __init__(self,store,improvement): self.store,self.improvement=store,improvement
    async def dispatch(self,request):
        if not isinstance(request,dict): raise FeatureError("invalid_request","request must be object")
        op=request.get("operation"); values={k:v for k,v in request.items() if k!="operation"}
        if op=="schemas.list" and not values: return self.store.schemas.list()
        if op=="feature.write": return self.store.write(**values)
        if op=="feature.label": return self.store.label(**values)
        if op=="feature.query": return self.store.query(**values)
        if op=="feature.delete": return self.store.delete(**values)
        if op=="dataset.get": return self.store.get_snapshot(**values)
        if op=="stats" and not values: return self.store.stats()
        if op=="dataset.snapshot":
            result=self.store.snapshot(**values)
            try:
                self.improvement.call(
                    "dataset.register",dataset_id=result["dataset_id"],
                    task=result["task"],item_count=result["item_count"],
                    schema_name=result["schema_name"],content_sha256=result["content_sha256"],
                    privacy=result["privacy"])
            except Exception as exc:
                # Do not leave an unregistered snapshot claiming training readiness.
                self.store.drop_snapshot(result["snapshot_id"])
                raise FeatureError("dataset_registration_failed","Improvement dataset authorization required") from exc
            return result
        raise FeatureError("unknown_operation","unknown operation")
    async def handle(self,reader,writer):
        try:
            raw=await reader.readline()
            if not raw or len(raw)>64*1024 or not raw.endswith(b"\n"): raise FeatureError("invalid_request","invalid framing")
            value={"ok":True,"result":await self.dispatch(json.loads(raw))}
        except FeatureError as exc: value={"ok":False,"code":exc.code,"error":str(exc)}
        except (json.JSONDecodeError,UnicodeDecodeError,TypeError,ValueError) as exc:
            value={"ok":False,"code":"invalid_arguments","error":str(exc)}
        except Exception as exc: value={"ok":False,"code":"internal_error","error":type(exc).__name__}
        writer.write((json.dumps(value,ensure_ascii=False,separators=(",", ":"))+"\n").encode()); await writer.drain()
        writer.close(); await writer.wait_closed()


async def run(socket_path,database):
    os.umask(0o077)
    path=Path(socket_path); path.parent.mkdir(parents=True,exist_ok=True); path.unlink(missing_ok=True)
    improvement=ImprovementClient(timeout=2)
    def consent(scope): return improvement.call("consent.status",scope=scope).get("allowed",False)
    store=FeatureStore(database,consent=consent); service=Service(store,improvement)
    server=await asyncio.start_unix_server(service.handle,path=socket_path); os.chmod(socket_path,0o600)
    try:
        async with server: await server.serve_forever()
    finally: store.close(); path.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--socket",default=DEFAULT_SOCKET)
    parser.add_argument("--database",default=str(DEFAULT_DB)); args=parser.parse_args()
    asyncio.run(run(args.socket,args.database))
if __name__=="__main__": main()
