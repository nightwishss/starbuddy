"""Owner-only API for explicit, auditable model improvement decisions."""

from __future__ import annotations

import argparse, asyncio, json, os
from pathlib import Path
from .client import DEFAULT_SOCKET
from .store import ImprovementError, ImprovementStore

DEFAULT_DB = Path.home() / ".local/state/vq-platform/improvement.sqlite3"

class Service:
    def __init__(self, store): self.store = store
    async def dispatch(self, request):
        if not isinstance(request, dict): raise ImprovementError("invalid_request", "request must be object")
        op=request.get("operation"); values={k:v for k,v in request.items() if k != "operation"}
        calls={"consent.set":self.store.set_consent,"consent.status":self.store.consent_status,
               "feedback.add":self.store.add_feedback,"dataset.register":self.store.register_dataset,
               "evaluation.record":self.store.record_evaluation,"evaluation.get":self.store.evaluation,
               "deployment.seed":self.store.seed_deployment,"deployment.promote":self.store.promote,
               "deployment.rollback":self.store.rollback,"deployment.get":self.store.deployment,
               "stats":self.store.stats}
        if op not in calls: raise ImprovementError("unknown_operation", "unknown operation")
        if op == "stats" and values: raise ImprovementError("invalid_arguments", "stats takes no arguments")
        return calls[op](**values)
    async def handle(self, reader, writer):
        try:
            raw=await reader.readline()
            if not raw or len(raw)>64*1024 or not raw.endswith(b"\n"): raise ImprovementError("invalid_request", "invalid framing")
            value={"ok":True,"result":await self.dispatch(json.loads(raw))}
        except ImprovementError as exc: value={"ok":False,"code":exc.code,"error":str(exc)}
        except Exception as exc: value={"ok":False,"code":"invalid_arguments","error":str(exc)}
        writer.write((json.dumps(value,ensure_ascii=False,separators=(",", ":"))+"\n").encode()); await writer.drain()
        writer.close(); await writer.wait_closed()

async def run(socket_path, database):
    os.umask(0o077)
    path=Path(socket_path); path.parent.mkdir(parents=True,exist_ok=True); path.unlink(missing_ok=True)
    store=ImprovementStore(database); server=await asyncio.start_unix_server(Service(store).handle,path=socket_path)
    os.chmod(socket_path,0o600)
    try:
        async with server: await server.serve_forever()
    finally: store.close(); path.unlink(missing_ok=True)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--socket",default=DEFAULT_SOCKET); p.add_argument("--database",default=str(DEFAULT_DB))
    a=p.parse_args(); asyncio.run(run(a.socket,a.database))
if __name__ == "__main__": main()
