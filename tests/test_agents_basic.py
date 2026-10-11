"""Tests for the shared tools, the system prompt and the plain agent. No model, no compiler
beyond make."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from tests.fake_model_server import FakeModelServer, finish_step, tool_call_step

from parallelise.agents import get_agent, tools
from parallelise.agents.base import AgentResult, build_system_prompt
from parallelise.config import Settings
from parallelise.model import ModelClient

pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")

MAKEFILE = "check:\n\ttest -f kernel.c\n"


def make_work_dir(root: Path) -> Path:
    """A work folder whose check target passes once kernel.c exists."""
    (root / "Makefile").write_text(MAKEFILE)
    notebook = {"cells": [{"cell_type": "markdown", "source": "# title"}]}
    (root / "n.ipynb").write_text(json.dumps(notebook))
    return root


def scripted_run(work_dir: Path, steps: list[dict], agent_name: str = "basic") -> tuple:
    with FakeModelServer(steps) as server:
        settings = Settings(gateway_url=server.url, model="local", request_timeout_s=30)
        with ModelClient(settings) as client:
            result = get_agent(agent_name).run(work_dir, "convert it", settings, client, 10)
        return result, server


def test_read_notebook_renders_cells(tmp_path):
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": "# title"},
            {"cell_type": "code", "source": ["x = 1\n", "print(x)\n"]},
        ]
    }
    (tmp_path / "n.ipynb").write_text(json.dumps(nb))
    text = tools.read_notebook(tmp_path, "n.ipynb")
    assert "# title" in text and "print(x)" in text and "cell 1 (code)" in text


def test_write_then_read_makes_folders(tmp_path):
    tools.write_file(tmp_path, "sub/k.c", "int main(void){return 0;}")
    assert tools.read_file(tmp_path, "sub/k.c").startswith("int main")


def test_paths_outside_the_work_folder_are_refused(tmp_path):
    assert tools.write_file(tmp_path, "../escape.txt", "x").startswith("error:")
    assert not (tmp_path.parent / "escape.txt").exists()


def test_missing_file_is_an_error_string_not_an_exception(tmp_path):
    assert tools.read_file(tmp_path, "absent.h").startswith("error:")


def test_command_allow_list(tmp_path):
    assert "exit 0" in tools.run_shell(tmp_path, "ls")
    assert "not allowed" in tools.run_shell(tmp_path, "rm -rf /")
    assert "not allowed" in tools.run_shell(tmp_path, "bash -c hi")
    # Without a shell, ';' and what follows are plain arguments to ls.
    tools.run_shell(tmp_path, "ls ; rm -rf /")
    assert tmp_path.exists()


def test_system_prompt_follows_the_work_folder(tmp_path):
    (tmp_path / "integrate.h").write_text("")
    (tmp_path / "reference.h").write_text("")
    (tmp_path / "Makefile").write_text("CC = mpicc\n")
    prompt = build_system_prompt(tmp_path)
    assert "integrate.c" in prompt and "Target MPI" in prompt


def test_unknown_agent_name_is_rejected():
    with pytest.raises(ValueError):
        get_agent("nonsense")


def test_basic_agent_runs_end_to_end(tmp_path):
    work = make_work_dir(tmp_path)
    steps = [
        tool_call_step("write_file", {"path": "kernel.c", "content": "/* kernel */"}, 100, 20),
        tool_call_step("run_shell", {"command": "make check"}, 150, 10),
        finish_step("done", 200, 5),
    ]
    result, server = scripted_run(work, steps)
    assert isinstance(result, AgentResult)
    assert result.finished
    assert (work / "kernel.c").read_text() == "/* kernel */"
    assert result.requests == 3 == len(server.requests)
    assert result.prompt_tokens == 450 == server.expected_prompt_tokens
    assert result.completion_tokens == 35 == server.expected_completion_tokens
    assert len(result.latencies_s) == 3


def test_failing_checks_are_fed_back_until_the_budget_ends(tmp_path):
    work = make_work_dir(tmp_path)  # kernel.c is never written, so the checks keep failing
    result, server = scripted_run(work, [finish_step("done", 10, 1)])
    assert not result.finished
    assert result.requests == 10
    last = server.requests[-1]["messages"][-1]
    assert last["role"] == "user" and "not passing" in last["content"]
