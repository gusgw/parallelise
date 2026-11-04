"""A scripted, OpenAI-compatible model server for tests.

It answers each chat request with the next entry of a script, in order, so that an agent can
be run from start to finish with no real model. Every answer carries a ``usage`` block, so
tests can check that the agents add up the token counts correctly.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
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


def slowly(step: dict, delay_s: float) -> dict:
    """Return a copy of a scripted step that the server answers only after ``delay_s`` seconds."""
    return {**step, "delay_s": delay_s}


class FakeModelServer:
    """Serves ``/v1/chat/completions`` from a list of scripted steps.

    After the script is used up, the last step is repeated. ``requests`` keeps every request
    body received, for inspection. A step may carry ``delay_s``, a number of seconds to wait
    before answering (see ``slowly``). The first ``throttle_first`` chat requests are refused
    with HTTP 429 (too many requests), with a ``Retry-After`` header of ``retry_after`` when that
    is given; they are counted in ``throttled`` and are not scripted steps. With
    ``fail_after`` set, every chat request after that many answered ones gets HTTP 400, which
    no client retries and so stands in for a model service that fails in the middle of a run.
    """

    def __init__(
        self,
        script: list[dict],
        host: str = "127.0.0.1",
        port: int = 0,
        throttle_first: int = 0,
        retry_after: str | None = None,
        fail_after: int | None = None,
    ) -> None:
        self.script = script
        self.requests: list[dict[str, Any]] = []
        self.throttled = 0
        self.refused = 0
        self._throttle_first = throttle_first
        self._retry_after = retry_after
        self._fail_after = fail_after
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: object) -> None:  # keep test output quiet
                pass

            def do_POST(self) -> None:  # noqa: N802 - name fixed by the base class
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                code, headers, payload = outer._respond(body)
                data = json.dumps(payload).encode()
                self.send_response(code)
                for name, value in headers.items():
                    self.send_header(name, value)
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

        self._httpd = HTTPServer((host, port), Handler)
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

    def _respond(self, body: dict) -> tuple[int, dict[str, str], dict]:
        """Return the status, the extra headers and the body for one chat request."""
        with self._lock:
            if self.throttled < self._throttle_first:
                self.throttled += 1
                headers = {"Retry-After": self._retry_after} if self._retry_after else {}
                return 429, headers, {"error": {"message": "slow down", "code": "429"}}
            if self._fail_after is not None and len(self.requests) >= self._fail_after:
                self.refused += 1
                return 400, {}, {"error": {"message": "the service refused", "code": "400"}}
        payload = self._next_answer(body)
        return 200, {}, payload

    def _next_answer(self, body: dict) -> dict:
        with self._lock:
            index = len(self.requests)
            self.requests.append(body)
        step = self.script[min(index, len(self.script) - 1)]
        if step.get("delay_s"):
            time.sleep(step["delay_s"])
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


def conversion_script(reference_dir: Path, name: str, backend: str = "openmp") -> list[dict]:
    """Return a script in which the model writes the known-good solution and then finishes.

    ``reference_dir`` holds ``<name>.c`` (``<name>.cu`` for CUDA) and ``test_<name>.c``, as a
    problem's ``<backend>/reference`` folder does. The model writes both files, runs
    ``make check``, and says it has finished.
    """
    routine = f"{name}.cu" if backend == "cuda" else f"{name}.c"
    test = f"test_{name}.c"
    return [
        tool_call_step(
            "write_file", {"path": routine, "content": (reference_dir / routine).read_text()},
            100, 20),
        tool_call_step(
            "write_file", {"path": test, "content": (reference_dir / test).read_text()}, 120, 20),
        tool_call_step("run_shell", {"command": "make check"}, 150, 10),
        finish_step("The checks pass, so I have finished.", 200, 5),
    ]


def main(argv: list[str] | None = None) -> int:
    """Serve a script until interrupted, for trying the command line without a real model.

    ``--script health`` answers every chat request with one plain sentence, which is all
    that ``parallelise health`` needs. ``--script convert`` answers a conversion by writing
    the known-good solution from ``--reference-dir``. The base URL to use as the gateway is
    printed on the first line of standard output, and also written to ``--url-file`` if given.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--script", choices=("health", "convert"), default="health")
    parser.add_argument("--reference-dir", type=Path, help="folder holding the known-good files")
    parser.add_argument("--name", default="integrate", help="the problem's name")
    parser.add_argument("--backend", default="openmp", choices=("openmp", "mpi", "cuda"))
    parser.add_argument("--host", default="127.0.0.1", help="address to listen on")
    parser.add_argument("--port", type=int, default=0, help="port to listen on (0: any free one)")
    parser.add_argument("--url-file", type=Path, help="write the base URL to this file")
    args = parser.parse_args(argv)
    if args.script == "convert":
        if args.reference_dir is None:
            parser.error("--script convert needs --reference-dir")
        script = conversion_script(args.reference_dir, args.name, args.backend)
    else:
        script = [finish_step("ready", 5, 1)]
    with FakeModelServer(script, host=args.host, port=args.port) as server:
        url = server.url
        print(url, flush=True)
        if args.url_file:
            args.url_file.write_text(url + "\n", encoding="utf-8")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
