"""Tests for the two framework agents, driven by the scripted fake model server."""

from __future__ import annotations

import shutil
import warnings

import pytest
from tests.fake_model_server import FakeModelServer, finish_step, slowly, tool_call_step
from tests.test_agents_basic import make_work_dir

from parallelise import model
from parallelise.agents import get_agent
from parallelise.config import Settings
from parallelise.model import ModelClient

pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")


def run_with_script(agent_name, work_dir, steps, budget=25, server_options=None, **settings):
    with FakeModelServer(steps, **(server_options or {})) as server:
        options = {"gateway_url": server.url, "model": "local", "request_timeout_s": 30, **settings}
        settings_ = Settings(**options)
        with ModelClient(settings_) as client:
            result = get_agent(agent_name).run(work_dir, "convert it", settings_, client, budget)
        return result, server


WRITE = tool_call_step("write_file", {"path": "routine.c", "content": "/* routine */"}, 100, 20)


def test_pydantic_agent_runs_end_to_end(tmp_path):
    pytest.importorskip("pydantic_ai")
    work = make_work_dir(tmp_path)
    steps = [WRITE, tool_call_step("run_checks", {}, 150, 10), finish_step("done", 200, 5)]
    result, server = run_with_script("pydantic", work, steps)
    assert result.finished, result.notes
    assert (work / "routine.c").exists()
    assert result.requests == 3 == len(server.requests)
    assert result.prompt_tokens == 450 == server.expected_prompt_tokens
    assert result.completion_tokens == 35 == server.expected_completion_tokens
    assert len(result.latencies_s) == 3


def test_pydantic_agent_refuses_to_finish_before_the_checks_pass(tmp_path):
    pytest.importorskip("pydantic_ai")
    work = make_work_dir(tmp_path)  # nothing is written, so run_checks is never green
    result, _ = run_with_script("pydantic", work, [finish_step("done", 10, 1)])
    assert not result.finished


def test_langgraph_agent_runs_end_to_end(tmp_path):
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_openai")
    work = make_work_dir(tmp_path)
    steps = [WRITE, tool_call_step("run_shell", {"command": "make check"}, 150, 10),
             finish_step("done", 200, 5)]
    result, server = run_with_script("langgraph", work, steps)
    assert result.finished, result.notes
    assert (work / "routine.c").exists()
    assert result.requests == 3 == len(server.requests)
    assert result.prompt_tokens == 450 == server.expected_prompt_tokens
    assert result.completion_tokens == 35 == server.expected_completion_tokens
    assert len(result.latencies_s) == 3


# --- usage counts survive a failed or interrupted run -------------------------------------------

READ = tool_call_step("read_file", {"path": "n.ipynb"}, 100, 10)


def needs(agent_name):
    if agent_name == "pydantic":
        pytest.importorskip("pydantic_ai")
    else:
        pytest.importorskip("langgraph")
        pytest.importorskip("langchain_openai")


AGENTS = ["pydantic", "langgraph"]


@pytest.mark.parametrize("agent_name", AGENTS)
def test_counts_are_kept_when_the_model_service_fails_part_way(tmp_path, agent_name):
    needs(agent_name)
    work = make_work_dir(tmp_path)
    result, server = run_with_script(
        agent_name, work, [READ], server_options={"fail_after": 2}
    )
    assert not result.finished and result.error  # the error is reported, not raised
    assert result.requests == 2 == len(server.requests)
    assert (result.prompt_tokens, result.completion_tokens) == (200, 20)
    assert len(result.latencies_s) == 2


def test_pydantic_counts_are_kept_when_the_step_budget_is_spent(tmp_path):
    needs("pydantic")
    work = make_work_dir(tmp_path)
    result, server = run_with_script("pydantic", work, [READ], budget=3)
    assert not result.finished
    assert result.requests == 3 == len(server.requests)
    assert result.prompt_tokens == 300 and result.completion_tokens == 30


def test_langgraph_counts_are_kept_when_the_step_limit_is_reached(tmp_path):
    needs("langgraph")
    work = make_work_dir(tmp_path)
    result, server = run_with_script("langgraph", work, [READ], budget=6)
    assert not result.finished and result.requests == len(server.requests) > 0
    assert result.prompt_tokens == 100 * result.requests


# --- the time limit ---------------------------------------------------------------------------


@pytest.mark.parametrize("agent_name", AGENTS)
def test_framework_agent_stops_between_steps_when_the_time_limit_passes(tmp_path, agent_name):
    needs(agent_name)
    work = make_work_dir(tmp_path)
    result, server = run_with_script(
        agent_name, work, [slowly(READ, 0.3)], budget=40, time_limit_s=0.5
    )
    assert result.time_limit_reached and not result.finished
    assert 1 <= result.requests <= 4 and result.requests == len(server.requests)
    assert result.prompt_tokens == 100 * result.requests


@pytest.mark.parametrize("agent_name", AGENTS)
def test_framework_agent_makes_no_call_when_the_time_is_already_spent(tmp_path, agent_name):
    needs(agent_name)
    work = make_work_dir(tmp_path)
    result, server = run_with_script(agent_name, work, [READ], time_limit_s=1e-9)
    assert result.time_limit_reached and result.requests == 0 and server.requests == []


# --- throttling ------------------------------------------------------------------------------


@pytest.fixture
def recorded_waits(monkeypatch):
    """Replace sleeping in the model module by writing each wait to a list."""
    waits: list[float] = []

    async def async_sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(model, "_sleep", waits.append)
    monkeypatch.setattr(model, "_async_sleep", async_sleep)
    return waits


@pytest.mark.parametrize("agent_name", AGENTS)
def test_framework_agent_waits_out_a_throttled_answer(tmp_path, recorded_waits, agent_name):
    needs(agent_name)
    work = make_work_dir(tmp_path)
    steps = [WRITE, tool_call_step("run_checks", {}, 150, 10) if agent_name == "pydantic"
             else tool_call_step("run_shell", {"command": "make check"}, 150, 10),
             finish_step("done", 200, 5)]
    result, server = run_with_script(
        agent_name, work, steps, server_options={"throttle_first": 2, "retry_after": "3"}
    )
    assert result.finished, result.notes
    assert server.throttled == 2 and recorded_waits == [3.0, 3.0]
    assert result.requests == 3 == len(server.requests)
    assert len(result.latencies_s) == 3  # the refused answers are not counted as model calls


@pytest.mark.parametrize("agent_name", AGENTS)
def test_framework_agent_gives_up_once_the_allowance_is_spent(tmp_path, recorded_waits, agent_name):
    needs(agent_name)
    work = make_work_dir(tmp_path)
    result, server = run_with_script(
        agent_name, work, [READ], throttle_wait_s=3,
        server_options={"throttle_first": 1000, "retry_after": "1"},
    )
    assert not result.finished and "429" in result.error and result.requests == 0
    assert recorded_waits == [1.0, 1.0, 1.0]
    # four answers in all: the first, and one after each wait. Neither framework's own retries
    # may repeat the sequence.
    assert server.throttled == 4


# --- the Pydantic AI client ---------------------------------------------------------------------


def deprecation_notices(caught):
    """Return the caught warnings that announce something removed in a later version."""
    messages = [str(w.message) for w in caught]
    return [m for m in messages if "v3" in m or "event loop" in m]


def test_pydantic_agent_builds_without_any_deprecation_warning(tmp_path):
    needs("pydantic")
    from parallelise.agents.pydantic_agent import build_agent

    work = make_work_dir(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_agent(work, Settings(gateway_url="http://127.0.0.1:9/v1", model="local"))
    assert deprecation_notices(caught) == []


def test_pydantic_agent_run_gives_no_deprecation_warning(tmp_path):
    needs("pydantic")
    work = make_work_dir(tmp_path)
    steps = [WRITE, tool_call_step("run_checks", {}, 150, 10), finish_step("done", 200, 5)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result, _ = run_with_script("pydantic", work, steps)
    assert result.finished
    assert deprecation_notices(caught) == []
