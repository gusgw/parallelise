"""Tests for the three agent designs, each driven by the scripted model server.

Every design gets the same toolset, whose check passes once routine.c exists, and reaches the
model through the client's meter, which counts every request whichever design made it.
"""

from __future__ import annotations

import importlib.util
import time

import pytest

from fake_model_server import FakeModelServer, finish_step, slowly, tool_call_step
from parallelise.agents import get_agent
from parallelise.agents.base import file_toolset
from parallelise.config import Settings
from parallelise.errors import ParalleliseError
from parallelise.model import ModelClient

DESIGNS = {"basic": None, "pydantic": "pydantic_ai", "langgraph": "langgraph"}
WRITE = tool_call_step("write_file", {"path": "routine.c", "content": "/* routine */"}, 100, 20)
READ = tool_call_step("read_file", {"path": "routine.c"}, 150, 10)
DONE = finish_step("done", 200, 5)


def run(design, folder, steps, budget=10, waits=None, server=None, **settings):
    """Run a design against a scripted model; return its result, its use and the server."""
    if DESIGNS[design] and importlib.util.find_spec(DESIGNS[design]) is None:
        pytest.skip(f"{DESIGNS[design]} is not installed")
    tools = file_toolset(folder, [], "routine.c", "Check routine.c.", lambda: "check passed",
                         lambda: (True, ""))
    with FakeModelServer(steps, **(server or {})) as fake:
        options = Settings(gateway_url=fake.url, model="local", request_timeout_s=30, **settings)
        with ModelClient(options, sleep=time.sleep if waits is None else waits.append) as client:
            if waits is not None:
                async def asleep(seconds):
                    waits.append(seconds)

                client.meter.async_sleep = asleep
            result = get_agent(design).run(folder, "convert it", options, client, budget, tools,
                                           "instructions")  # fmt: skip
            used = client.meter.summary()
    return result, used, fake


@pytest.mark.parametrize("design", DESIGNS)
def test_each_design_writes_checks_and_finishes_and_every_call_is_counted(tmp_path, design):
    result, used, server = run(design, tmp_path, [WRITE, READ, DONE])
    assert result.finished, result
    assert used["requests"] == len(server.requests) == 3
    assert used["prompt_tokens"] == server.expected_prompt_tokens


@pytest.mark.parametrize("design", DESIGNS)
def test_no_design_can_finish_before_the_check_passes(tmp_path, design):
    result, used, server = run(design, tmp_path, [DONE], budget=3)
    assert not result.finished and len(server.requests) > 1


@pytest.mark.parametrize("design", DESIGNS)
def test_the_time_limit_stops_each_design_between_steps(tmp_path, design):
    result, used, _ = run(design, tmp_path, [slowly(WRITE, 1.5), READ, DONE], time_limit_s=1)
    assert result.time_limit_reached and not result.finished and used["requests"] < 3


@pytest.mark.parametrize("design", DESIGNS)
def test_a_model_that_fails_part_way_ends_the_run_and_its_calls_are_still_counted(tmp_path,
                                                                                   design):
    result, used, _ = run(design, tmp_path, [WRITE, READ, DONE], server={"fail_after": 1})
    assert not result.finished and "400" in result.error and used["requests"] == 1


@pytest.mark.parametrize("design", DESIGNS)
def test_each_design_waits_out_refusals_through_the_meter(tmp_path, design):
    waits: list[float] = []
    result, used, _ = run(design, tmp_path, [WRITE, READ, DONE], waits=waits,
                          server={"statuses": [429, 429], "retry_after": "3"})
    assert result.finished and waits == [10.0, 10.0] and used["requests"] == 3


@pytest.mark.parametrize("design", DESIGNS)
def test_waiting_stops_once_the_allowance_is_spent_and_is_not_repeated(tmp_path, design):
    waits: list[float] = []
    result, used, server = run(design, tmp_path, [READ], waits=waits, throttle_wait_s=25,
                               server={"statuses": [429] * 100})
    assert not result.finished and "429" in result.error and used["requests"] == 0
    # Three answers in all: the first, and one after each pause; no library may begin again.
    assert waits == [10.0, 10.0] and server.refused == 3


def test_an_unknown_design_is_refused():
    with pytest.raises(ParalleliseError, match="not one of"):
        get_agent("magic")
