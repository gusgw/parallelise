"""Tests for the model client, using a local fake HTTP server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from parallelise.config import Settings
from parallelise.errors import ModelError
from parallelise.model import ModelClient


class Fake:
    """Scripted behaviour for the fake server: a list of status codes, then success."""

    def __init__(self, statuses=()):
        self.statuses = list(statuses)
        self.posts = 0
        self.last_auth = None
        self.last_body = None
        self.models = ["local", "other"]


@pytest.fixture
def server():
    fake = Fake()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            fake.last_auth = self.headers.get("Authorization")
            if self.path.endswith("/models"):
                self._send(200, {"data": [{"id": m} for m in fake.models]})
            else:
                self._send(404, {})

        def do_POST(self):
            fake.posts += 1
            fake.last_auth = self.headers.get("Authorization")
            fake.last_body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if fake.statuses:
                code = fake.statuses.pop(0)
                if code != 200:
                    self._send(code, {"error": "scripted"})
                    return
            self._send(200, {
                "choices": [{"message": {"role": "assistant", "content": None}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            })

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    fake.url = f"http://127.0.0.1:{httpd.server_port}/v1"
    yield fake
    httpd.shutdown()
    httpd.server_close()


def client_for(server, **kw):
    overrides = {"gateway_url": server.url, "request_timeout_s": 5, **kw}
    settings = Settings.from_env(overrides, environ={})
    return ModelClient(settings, retry_delay_s=0)


def test_chat_records_usage_and_sends_key(server):
    with client_for(server, api_key="secret") as client:
        result = client.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
        assert result.message["content"] == ""  # None is turned into an empty string
        assert (result.prompt_tokens, result.completion_tokens) == (10, 5)
        client.chat([{"role": "user", "content": "again"}])
        s = client.summary()
    assert s["requests"] == 2
    assert s["total_tokens"] == 30
    assert s["completion_tokens"] == 10
    assert s["completion_tokens_per_second"] > 0
    assert s["median_latency_s"] > 0 and s["mean_latency_s"] > 0
    assert server.last_auth == "Bearer secret"
    assert server.last_body["model"] == "local"
    assert "tools" not in server.last_body


def test_server_errors_are_retried(server):
    server.statuses = [503, 502]
    with client_for(server) as client:
        client.chat([{"role": "user", "content": "x"}])
        s = client.summary()
    assert server.posts == 3
    assert s["requests"] == 1 and s["failed_requests"] == 2


def test_gives_up_after_three_attempts(server):
    server.statuses = [500, 500, 500, 500]
    with client_for(server) as client, pytest.raises(ModelError, match="3 attempts"):
        client.chat([{"role": "user", "content": "x"}])
    assert server.posts == 3


def test_client_errors_are_not_retried(server):
    server.statuses = [400]
    with client_for(server) as client, pytest.raises(ModelError, match="HTTP 400"):
        client.chat([{"role": "user", "content": "x"}])
    assert server.posts == 1


def test_connection_refused_is_retried_then_reported():
    overrides = {"gateway_url": "http://127.0.0.1:9/v1", "request_timeout_s": 2}
    settings = Settings.from_env(overrides, environ={})
    with ModelClient(settings, retry_delay_s=0) as client:
        with pytest.raises(ModelError, match="3 attempts"):
            client.chat([{"role": "user", "content": "x"}])
        report = client.health_check()
    assert not report.reachable and not report.healthy
    assert "Could not reach" in report.detail


def test_health_check_healthy(server):
    with client_for(server) as client:
        report = client.health_check()
        assert client.calls == []  # the test call is not counted
    assert report.healthy and report.models == ["local", "other"]
    assert report.round_trip_s > 0


def test_health_check_model_missing(server):
    with client_for(server, model="absent") as client:
        report = client.health_check()
    assert report.reachable and not report.model_listed and not report.healthy
    assert "absent" in report.detail
