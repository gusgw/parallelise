"""Tests for the model client and its meter: retries, waiting out refusals, and counting."""

import asyncio

import httpx
import httpx2
import pytest

from fake_model_server import FakeModelServer, finish_step
from parallelise.config import Settings
from parallelise.errors import ModelError
from parallelise.model import ModelClient, Throttle


def client_for(server, waits, **settings):
    values = {"gateway_url": server.url if server else "http://127.0.0.1:9/v1", "model": "local",
              "api_key": "k", **settings}  # fmt: skip
    return ModelClient(Settings(**values), retry_delay_s=0, sleep=waits.append)


def test_a_chat_returns_the_message_and_the_meter_counts_it():
    with FakeModelServer([finish_step("hello", 7, 3)]) as server:
        with client_for(server, []) as client:
            assert client.chat([{"role": "user", "content": "hi"}])["content"] == "hello"
            used = client.meter.summary()
    assert (used["requests"], used["prompt_tokens"], used["completion_tokens"]) == (1, 7, 3)


@pytest.mark.parametrize(("statuses", "answered"), [([500], True), ([500, 502, 503], False),
                                                    ([400], False)])  # fmt: skip
def test_server_errors_are_tried_again_three_times_and_client_errors_not(statuses, answered):
    with FakeModelServer([finish_step("ok", 1, 1)], statuses=statuses) as server:
        with client_for(server, []) as client:
            if answered:
                client.chat([])
            else:
                with pytest.raises(ModelError):
                    client.chat([])
        assert server.refused == len(statuses)


def test_an_endpoint_that_cannot_be_reached_is_reported_after_three_attempts():
    with client_for(None, []) as client, pytest.raises(ModelError, match="3 attempts"):
        client.chat([])


def test_refusals_are_waited_out_for_the_time_named_and_not_counted():
    waits: list[float] = []
    with FakeModelServer([finish_step("ok", 1, 1)], statuses=[429, 429], retry_after="3") as server:
        with client_for(server, waits) as client:
            client.chat([])
            assert client.meter.summary()["requests"] == 1
    assert waits == [3.0, 3.0]


def test_waiting_stops_when_the_allowance_is_spent():
    waits: list[float] = []
    with FakeModelServer([finish_step("ok", 1, 1)], statuses=[429] * 10) as server:
        with client_for(server, waits, throttle_wait_s=10) as client:
            with pytest.raises(ModelError, match="429"):
                client.chat([])
    assert sum(waits) == 10


@pytest.mark.parametrize(("told", "allowance", "first"), [
    (None, 120, 2.0), ("5", 120, 5.0), ("0", 120, 1.0), ("soon", 120, 2.0), ("300", 120, None),
    (None, 0, None),
])  # fmt: skip
def test_the_throttle_obeys_retry_after_within_its_allowance(told, allowance, first):
    assert Throttle(allowance).next_wait(told) == first


def test_the_throttle_doubles_its_wait_up_to_thirty_seconds():
    throttle = Throttle(1000)
    assert [throttle.next_wait(None) for _ in range(6)] == [2, 4, 8, 16, 30, 30]


@pytest.mark.parametrize("library", [httpx, httpx2], ids=["httpx", "httpx2"])
def test_asynchronous_clients_wait_and_are_counted_by_the_same_meter(library):
    waits: list[float] = []

    async def asleep(seconds):
        waits.append(seconds)

    with FakeModelServer([finish_step("ok", 4, 2)], statuses=[429]) as server:
        with client_for(server, []) as client:
            client.meter.async_sleep = asleep

            async def ask():
                transport = client.meter.async_transport(library)
                async with library.AsyncClient(transport=transport) as c:
                    await c.post(f"{server.url}/chat/completions", json={"model": "local"})

            asyncio.run(ask())
            assert client.meter.summary()["completion_tokens"] == 2
    assert waits == [2.0]


def test_the_endpoint_lists_its_models_or_says_it_cannot_be_reached():
    with FakeModelServer([finish_step("ok", 1, 1)]) as server, client_for(server, []) as client:
        assert client.models() == ["local"]
    with client_for(None, []) as client, pytest.raises(ModelError, match="Could not reach"):
        client.models()
