"""Owner-only Unix API for trusted application feedback actions."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from feature_store_service import FeatureStoreClient, FeatureStoreClientError
from improvement_service import ImprovementClient, ImprovementClientError
from posture_service import PostureClient, PostureClientError

from .client import DEFAULT_SOCKET
from .collector import FeedbackCollector, FeedbackError


class Service:
    def __init__(self,collector): self.collector=collector
    async def dispatch(self,request):
        if not isinstance(request,dict): raise FeedbackError("invalid_request","request must be object")
        op=request.get("operation"); values={k:v for k,v in request.items() if k!="operation"}
        calls={"posture.label_current":self.collector.label_current_posture,
               "vision.label_summary":self.collector.label_vision_summary,
               "conversation.rate":self.collector.rate_conversation,
               "stats":self.collector.stats}
        if op not in calls: raise FeedbackError("unknown_operation","unknown operation")
        if op=="stats" and values: raise FeedbackError("invalid_arguments","stats takes no arguments")
        return calls[op](**values)
    async def handle(self,reader,writer):
        try:
            raw=await reader.readline()
            if not raw or len(raw)>32*1024 or not raw.endswith(b"\n"):
                raise FeedbackError("invalid_request","invalid framing")
            value={"ok":True,"result":await self.dispatch(json.loads(raw))}
        except (FeedbackError,FeatureStoreClientError,ImprovementClientError,PostureClientError) as exc:
            value={"ok":False,"code":getattr(exc,"code","dependency_error"),"error":str(exc)}
        except (json.JSONDecodeError,UnicodeDecodeError,TypeError,ValueError) as exc:
            value={"ok":False,"code":"invalid_arguments","error":str(exc)}
        except Exception as exc:
            value={"ok":False,"code":"internal_error","error":type(exc).__name__}
        writer.write((json.dumps(value,ensure_ascii=False,separators=(",", ":"))+"\n").encode())
        await writer.drain(); writer.close(); await writer.wait_closed()


async def run(socket_path):
    os.umask(0o077); path=Path(socket_path); path.parent.mkdir(parents=True,exist_ok=True)
    path.unlink(missing_ok=True)
    collector=FeedbackCollector(FeatureStoreClient(),ImprovementClient(),PostureClient())
    server=await asyncio.start_unix_server(Service(collector).handle,path=socket_path); os.chmod(socket_path,0o600)
    try:
        async with server: await server.serve_forever()
    finally: path.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--socket",default=DEFAULT_SOCKET)
    args=parser.parse_args(); asyncio.run(run(args.socket))
if __name__=="__main__": main()
