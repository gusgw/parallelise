"""Tests for agent_pydantic.agent — construction + TestModel runs (no network)."""
import pytest
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel

from agent_pydantic import agent


def test_build_model_is_openai_chat(monkeypatch):
    monkeypatch.setenv("MODEL_BASE_URL", "http://x/v1")
    monkeypatch.setenv("MODEL_NAME", "foo")
    monkeypatch.setenv("MODEL_API_KEY", "")
    assert isinstance(agent.build_model(), OpenAIChatModel)


def test_all_variants_build(tmp_path):
    for build in (agent.build_agent, agent.build_min_agent,
                  agent.build_typed_agent, agent.build_typed_checked_agent):
        assert isinstance(build(str(tmp_path)), Agent)


def test_min_variant_drives_a_tool(tmp_path):
    (tmp_path / "a.txt").write_text("hi")
    a = agent.build_min_agent(str(tmp_path))   # no forcing/validator -> finishes freely
    with a.override(model=TestModel(call_tools=["ls"])):
        result = a.run_sync("list the files")
    assert result.output is not None


def test_full_variant_finishes_once_tests_pass(tmp_path):
    # run_tests wired to a passing command -> tests_passed becomes True -> the
    # must-pass validator allows the run to finish.
    a = agent.build_agent(str(tmp_path), test_cmd=["python", "-c", "print('ok')"])
    with a.override(model=TestModel(call_tools=["run_tests"])):
        result = a.run_sync("fix it", deps=agent.RunState())
    assert result.output is not None


def test_typed_variant_returns_structured_report(tmp_path):
    a = agent.build_typed_agent(str(tmp_path), test_cmd=["python", "-c", "print('ok')"])
    with a.override(model=TestModel(call_tools=["run_tests"])):
        result = a.run_sync("fix it")
    assert isinstance(result.output, agent.FixReport)
    assert isinstance(result.output.tests_pass, bool)


def test_typed_checked_accepts_report_once_tests_pass(tmp_path):
    # passing test_cmd -> run_tests sets tests_passed -> the validator accepts the
    # structured report, and the claim is corrected to match reality.
    a = agent.build_typed_checked_agent(str(tmp_path), test_cmd=["python", "-c", "print('ok')"])
    with a.override(model=TestModel(call_tools=["run_tests"])):
        result = a.run_sync("fix it", deps=agent.RunState())
    assert isinstance(result.output, agent.FixReport)
    assert result.output.tests_pass is True


def test_typed_checked_refuses_report_while_tests_fail(tmp_path):
    # failing test_cmd -> the validator rejects every finish attempt; with no way to
    # pass, the output-retry budget is exhausted rather than a false report accepted.
    a = agent.build_typed_checked_agent(str(tmp_path), test_cmd=["python", "-c", "import sys; sys.exit(1)"])
    with a.override(model=TestModel(call_tools=["run_tests"])):
        # Either way the run exhausts a budget rather than ever accepting a false
        # report: output-retries (UnexpectedModelBehavior) or requests (UsageLimitExceeded).
        with pytest.raises((UnexpectedModelBehavior, UsageLimitExceeded)):
            a.run_sync("fix it", deps=agent.RunState())


def test_oversized_edit_rejected():
    agent.check_snippet_size("short", "short")   # fine
    with pytest.raises(ModelRetry, match="too large"):
        agent.check_snippet_size("x" * (agent.MAX_EDIT_SNIPPET + 1), "y")
    with pytest.raises(ModelRetry, match="too large"):
        agent.check_snippet_size("x", "y" * (agent.MAX_EDIT_SNIPPET + 1))


def test_no_repeat_guard_nudges_on_exact_repeat():
    no_repeat = agent._make_no_repeat()
    assert no_repeat("grep", {"pattern": "x"}) is None      # first call: fine
    nudge = no_repeat("grep", {"pattern": "x"})             # exact repeat: nudged
    assert nudge is not None and "already made" in nudge
    assert no_repeat("grep", {"pattern": "y"}) is None      # different args: fine again
