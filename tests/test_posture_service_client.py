import json, threading, unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from posture_service import PostureClient

class H(BaseHTTPRequestHandler):
    bodies=[]
    def log_message(self,*a): pass
    def do_GET(self): self.send()
    def do_POST(self):
        self.__class__.bodies.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length","0"))) or b"{}")); self.send()
    def send(self):
        body=json.dumps({"state":"healthy","enabled":True}).encode(); self.send_response(200)
        self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)

class PostureClientTests(unittest.TestCase):
    def test_all_operations_use_service_contract(self):
        server=ThreadingHTTPServer(("127.0.0.1",0),H); thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            client=PostureClient(f"http://127.0.0.1:{server.server_port}/api/posture")
            self.assertEqual(client.status()["state"],"healthy")
            for operation in (client.start,client.pause,client.calibrate,client.reset,client.simulate):
                self.assertTrue(operation()["enabled"])
            client.monitor("task_"+"a"*32,duration_seconds=600)
            self.assertEqual(H.bodies[-1]["duration_seconds"],600)
        finally: server.shutdown(); server.server_close()

if __name__ == "__main__": unittest.main()
