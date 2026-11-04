"""Tests for the HTTP clients that wait out throttling for the two framework agents.

The clients are driven here by a stand-in transport, so no network and no real waiting.
"""

import asyncio

import httpx
import pytest

from parallelise.agents.base import LatencyRecorder
from parallelise.model import ThrottledAsyncClient, ThrottledClient, throttled_async_client_class

URL = "http://gateway.test/v1/chat/completions"


def answers(*responses):
    """A transport that gives the listed responses in order, and counts the requests."""
    queue = list(responses)
    seen = []

    def handler(request):
        seen.append(request)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return httpx.MockTransport(handler), seen


def throttled(retry_after=None):
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return httpx.Response(429, headers=headers, json={"error": "slow down"})


def ok():
    return httpx.Response(200, json={"choices": []})


def test_client_waits_for_the_time_named_and_sends_the_same_request_again():
    transport, seen = answers(throttled("3"), ok())
    waits = []
    with ThrottledClient(throttle_wait_s=120, sleep=waits.append, transport=transport) as client:
        response = client.post(URL, json={"model": "m"})
    assert response.status_code == 200 and waits == [3.0]
    assert len(seen) == 2 and seen[0].content == seen[1].content != b""


def test_client_backs_off_when_no_time_is_named():
    transport, _ = answers(throttled(), throttled(), throttled(), ok())
    waits = []
    with ThrottledClient(throttle_wait_s=120, sleep=waits.append, transport=transport) as client:
        assert client.post(URL, json={}).status_code == 200
    assert waits == [2.0, 4.0, 8.0]


def test_client_returns_the_last_answer_when_the_allowance_is_spent():
    transport, seen = answers(throttled("1"))
    waits = []
    with ThrottledClient(throttle_wait_s=2, sleep=waits.append, transport=transport) as client:
        response = client.post(URL, json={})
    assert response.status_code == 429 and waits == [1.0, 1.0] and len(seen) == 3
    # The OpenAI library underneath both frameworks must not repeat a sequence that has
    # already waited as long as the settings allow.
    assert response.headers["x-should-retry"] == "false"


def test_client_leaves_other_answers_alone():
    for status in (200, 400, 401, 500, 503):
        transport, seen = answers(httpx.Response(status, json={}))
        waits = []
        with ThrottledClient(throttle_wait_s=120, sleep=waits.append, transport=transport) as c:
            assert c.post(URL, json={}).status_code == status
        assert waits == [] and len(seen) == 1


def test_async_client_waits_and_retries():
    transport, seen = answers(throttled("5"), ok())
    waits = []

    async def sleep(seconds):
        waits.append(seconds)

    async def go():
        async with ThrottledAsyncClient(
            throttle_wait_s=120, sleep=sleep, transport=transport
        ) as client:
            return await client.post(URL, json={"model": "m"})

    response = asyncio.run(go())
    assert response.status_code == 200 and waits == [5.0] and len(seen) == 2


def test_async_client_class_can_be_made_for_another_http_library():
    httpx2 = pytest.importorskip("httpx2")
    client_class = throttled_async_client_class(httpx2.AsyncClient)
    assert issubclass(client_class, httpx2.AsyncClient)
    queue = [httpx2.Response(429, headers={"Retry-After": "2"}, json={}),
             httpx2.Response(200, json={})]
    transport = httpx2.MockTransport(lambda request: queue.pop(0))
    waits = []

    async def sleep(seconds):
        waits.append(seconds)

    async def go():
        async with client_class(throttle_wait_s=120, sleep=sleep, transport=transport) as c:
            return await c.post(URL, json={})

    assert asyncio.run(go()).status_code == 200 and waits == [2.0]


def test_latency_is_timed_for_each_attempt_and_only_good_answers_count():
    recorder = LatencyRecorder()
    transport, _ = answers(throttled("1"), throttled("1"), ok())
    with ThrottledClient(
        throttle_wait_s=120, sleep=lambda s: None, transport=transport,
        event_hooks=recorder.sync_hooks(),
    ) as client:
        client.post(URL, json={})
    assert len(recorder.latencies_s) == 1
