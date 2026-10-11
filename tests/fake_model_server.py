"""A scripted, OpenAI-compatible model server for tests.

It answers each chat request with the next entry of a script, in order, so that an agent can
be run from start to finish with no real model. Every answer carries a ``usage`` block, so
tests can check that the agents add up the token counts correctly.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


def tool_call_step(name: str, arguments: dict, prompt_tokens: int, completion_tokens: int) -> dict:
    """A scripted answer in which the model asks for one tool call."""
    return {
        "tool": (name, arguments),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def finish_step(text: str, prompt_tokens: int, completion_tokens: int) -> dict:
    """A scripted answer in which the model replies in plain words and stops."""
    return {"text": text, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


class FakeModelServer:
    """Serves ``/v1/chat/completions`` from a list of scripted steps.

    After the script is used up, the last step is repeated. ``requests`` keeps every request
    body received, for inspection.
    """

    def __init__(self, script: list[dict]) -> None:
        self.script = script
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # keep test output quiet
                pass

            def do_POST(self) -> None:  # noqa: N802 - name fixed by the base class
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                payload = outer._next_answer(body)
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802
                data = json.dumps({"data": [{"id": "local"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        """The base URL to use as the gateway URL."""
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    def __enter__(self) -> FakeModelServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    @property
    def expected_prompt_tokens(self) -> int:
        """Prompt tokens summed over the steps actually served."""
        return sum(s["prompt_tokens"] for s in self._served())

    @property
    def expected_completion_tokens(self) -> int:
        """Completion tokens summed over the steps actually served."""
        return sum(s["completion_tokens"] for s in self._served())

    def _served(self) -> list[dict]:
        count = len(self.requests)
        return [self.script[min(i, len(self.script) - 1)] for i in range(count)]

    def _next_answer(self, body: dict) -> dict:
        with self._lock:
            index = len(self.requests)
            self.requests.append(body)
        step = self.script[min(index, len(self.script) - 1)]
        message: dict[str, Any] = {"role": "assistant"}
        finish = "stop"
        if "tool" in step:
            name, arguments = step["tool"]
            message["content"] = None
            message["tool_calls"] = [
                {
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ]
            finish = "tool_calls"
        else:
            message["content"] = step["text"]
        return {
            "id": f"chatcmpl-{index}",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "local"),
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {
                "prompt_tokens": step["prompt_tokens"],
                "completion_tokens": step["completion_tokens"],
                "total_tokens": step["prompt_tokens"] + step["completion_tokens"],
            },
        }
