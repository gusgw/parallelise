"""Offline tests for the Pydantic-AI starter (TestModel — no endpoint needed).

The interesting one runs the agent against TestModel with a *passing* test
command, so the verification gate is satisfied and the run finishes — proving the
gate is wired correctly.

    cd nano_pydantic && python -m pytest -q
"""
from pydantic_ai.models.test import TestModel

import pyd_agent as pa


def test_build_agent_constructs(tmp_path):
    assert pa.build_agent(str(tmp_path)) is not None


def test_finishes_once_tests_pass(tmp_path):
    # run_tests wired to a command that always succeeds -> gate opens -> run ends
    agent = pa.build_agent(str(tmp_path), test_cmd=("python", "-c", "print('ok')"))
    with agent.override(model=TestModel(call_tools=["run_tests"])):
        result = agent.run_sync("fix it", deps=pa.State())
    assert result.output is not None


def test_state_defaults_red():
    assert pa.State().tests_passed is False
