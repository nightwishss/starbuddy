"""Loopback client for the persistent local Agent worker."""

import json
import urllib.error
import urllib.request


class LocalAgentError(RuntimeError):
    pass


class LocalAgentClient:
    def __init__(self, base_url="http://127.0.0.1:9003", timeout=45.0):
        if not base_url.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError("local Agent endpoint must use loopback")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, message):
        request = urllib.request.Request(
            self.base_url + "/v1/chat",
            data=json.dumps({"message": message}, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read(1024 * 1024))
            answer = payload["answer"].strip()
        except (OSError, urllib.error.URLError, KeyError, ValueError, json.JSONDecodeError) as exc:
            raise LocalAgentError("板端本地Agent暂时不可用，请稍后重试。") from exc
        if not answer:
            raise LocalAgentError("板端本地Agent没有返回内容。")
        return answer
