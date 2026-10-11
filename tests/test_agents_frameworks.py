"""Tests for the two framework agents, driven by the scripted fake model server."""

from __future__ import annotations

import shutil

import pytest
from tests.fake_model_server import FakeModelServer, finish_step, tool_call_step
from tests.test_agents_basic import make_work_dir

from parallelise.agents import get_agent
from parallelise.config import Settings
from parallelise.model import ModelClient

pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")


def run_with_script(agent_name, work_dir, steps):
    with FakeModelServer(steps) as server:
        settings = Settings(gateway_url=server.url, model="local", request_timeout_s=30)
        with ModelClient(settings) as client:
            result = get_agent(agent_name).run(work_dir, "convert it", settings, client, 25)
        return result, server


WRITE = tool_call_step("write_file", {"path": "kernel.c", "content": "/* kernel */"}, 100, 20)


def test_pydantic_agent_runs_end_to_end(tmp_path):
    pytest.importorskip("pydantic_ai")
    work = make_work_dir(tmp_path)
    steps = [WRITE, tool_call_step("check_gate", {}, 150, 10), finish_step("done", 200, 5)]
    result, server = run_with_script("pydantic", work, steps)
    assert result.finished, result.notes
    assert (work / "kernel.c").exists()
    assert result.requests == 3 == len(server.requests)
    assert result.prompt_tokens == 450 == server.expected_prompt_tokens
    assert result.completion_tokens == 35 == server.expected_completion_tokens
    assert len(result.latencies_s) == 3


def test_pydantic_agent_refuses_to_finish_before_the_checks_pass(tmp_path):
    pytest.importorskip("pydantic_ai")
    work = make_work_dir(tmp_path)  # nothing is written, so check_gate is never green
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
    assert (work / "kernel.c").exists()
    assert result.requests == 3 == len(server.requests)
    assert result.prompt_tokens == 450 == server.expected_prompt_tokens
    assert result.completion_tokens == 35 == server.expected_completion_tokens
    assert len(result.latencies_s) == 3
