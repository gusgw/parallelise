"""Tests for the model client, using a local fake HTTP server."""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from parallelise.config import Settings
from parallelise.errors import ModelError
from parallelise.model import ModelClient, Throttle, parse_retry_after


class Fake:
    """Scripted behaviour for the fake server: a list of status codes, then success.

    An entry of the list is a status code, or a pair of a status code and a dictionary of
    extra headers for that answer.
    """

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
        def _send(self, code, payload, headers=None):
            data = json.dumps(payload).encode()
            self.send_response(code)
            for name, value in (headers or {}).items():
                self.send_header(name, value)
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
                entry = fake.statuses.pop(0)
                code, headers = entry if isinstance(entry, tuple) else (entry, {})
                if code != 200:
                    self._send(code, {"error": "scripted"}, headers)
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


def client_for(server, waits=None, **kw):
    """A client for the fake server. Waits are written to ``waits`` instead of being slept."""
    overrides = {"gateway_url": server.url, "request_timeout_s": 5, **kw}
    settings = Settings.from_env(overrides, environ={})
    if waits is None:
        return ModelClient(settings, retry_delay_s=0)
    return ModelClient(settings, retry_delay_s=0, sleep=waits.append)


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


# --- throttling: HTTP 429 is waited out, not given up on --------------------------------------


def test_throttled_request_is_retried_after_the_time_the_server_names(server):
    server.statuses = [(429, {"Retry-After": "7"})]
    waits = []
    with client_for(server, waits) as client:
        client.chat([{"role": "user", "content": "x"}])
        s = client.summary()
    assert waits == [7.0]
    assert server.posts == 2
    assert s["requests"] == 1 and s["failed_requests"] == 1


def test_throttling_without_retry_after_waits_longer_each_time(server):
    server.statuses = [429, 429, 429]
    waits = []
    with client_for(server, waits) as client:
        client.chat([{"role": "user", "content": "x"}])
    assert waits == [2.0, 4.0, 8.0]
    assert server.posts == 4


def test_waiting_stops_when_the_allowance_is_spent(server):
    server.statuses = [429] * 20
    waits = []
    with client_for(server, waits, throttle_wait_s=10) as client:
        with pytest.raises(ModelError) as raised:
            client.chat([{"role": "user", "content": "x"}])
    assert waits == [2.0, 4.0, 4.0]  # the last wait is cut to what was left of the 10 seconds
    assert server.posts == 4
    message = str(raised.value)
    assert "429" in message and "10 seconds" in message and "PARALLELISE_THROTTLE_WAIT" in message


def test_a_retry_after_longer_than_the_allowance_gives_up_at_once(server):
    server.statuses = [(429, {"Retry-After": "500"})] * 3
    waits = []
    with client_for(server, waits) as client, pytest.raises(ModelError, match="429"):
        client.chat([{"role": "user", "content": "x"}])
    assert waits == [] and server.posts == 1


def test_a_retry_after_that_is_not_a_number_of_seconds_is_ignored(server):
    server.statuses = [(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})]
    waits = []
    with client_for(server, waits) as client:
        client.chat([{"role": "user", "content": "x"}])
    assert waits == [2.0]


def test_a_retry_after_of_zero_still_waits_a_little(server):
    server.statuses = [(429, {"Retry-After": "0"})]
    waits = []
    with client_for(server, waits) as client:
        client.chat([{"role": "user", "content": "x"}])
    assert waits == [1.0]


def test_an_allowance_of_zero_means_no_waiting(server):
    server.statuses = [429]
    waits = []
    with client_for(server, waits, throttle_wait_s=0) as client, pytest.raises(ModelError):
        client.chat([{"role": "user", "content": "x"}])
    assert waits == [] and server.posts == 1


def test_every_wait_is_logged(server, caplog):
    server.statuses = [(429, {"Retry-After": "7"}), 429]
    with caplog.at_level(logging.WARNING, logger="parallelise.model"):
        with client_for(server, []) as client:
            client.chat([{"role": "user", "content": "x"}])
    waiting = [r.getMessage() for r in caplog.records if "429" in r.getMessage()]
    assert len(waiting) == 2
    assert "7 seconds" in waiting[0] and "4 seconds" in waiting[1]


def test_throttling_does_not_use_up_the_attempts_for_other_failures(server):
    server.statuses = [429, 429, 429, 503, 503]
    waits = []
    with client_for(server, waits) as client:
        client.chat([{"role": "user", "content": "x"}])
    assert server.posts == 6
    assert len([w for w in waits if w > 0]) == 3  # only the three throttled answers waited


def test_other_client_errors_are_still_not_retried_while_throttled_before(server):
    server.statuses = [429, 401]
    waits = []
    with client_for(server, waits) as client, pytest.raises(ModelError, match="HTTP 401"):
        client.chat([{"role": "user", "content": "x"}])
    assert server.posts == 2


def test_default_allowance_is_two_minutes():
    assert Settings.from_env(environ={}).throttle_wait_s == 120


def test_retry_after_is_read_as_seconds_only():
    assert parse_retry_after("7") == 7.0
    assert parse_retry_after(" 2.5 ") == 2.5
    for unusable in (None, "", "soon", "-3", "nan", "inf", "Wed, 21 Oct 2026 07:28:00 GMT"):
        assert parse_retry_after(unusable) is None


def test_the_allowance_is_shared_by_the_answers_to_one_request():
    throttle = Throttle(allowance_s=20)
    assert throttle.next_wait(None) == 2.0
    assert throttle.next_wait("5") == 5.0
    assert throttle.next_wait(None) == 8.0  # the third answer: 2 * 2 * 2
    assert throttle.waited_s == 15.0
    assert throttle.next_wait("6") is None  # 6 seconds would pass the 5 left
    assert throttle.next_wait(None) == 5.0  # cut to what is left
    assert throttle.next_wait(None) is None
