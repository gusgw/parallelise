"""Tests for the command line: exit codes, messages, and the run flow with stand-in parts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from parallelise import cli
from parallelise.agents.base import AgentResult
from parallelise.results import ResultsStore

GOOD = (
    "N = 1000\nterms = [1.0 / (i + 1) for i in range(N)]\n",
    "ORACLE = {'kernel': 'harmonic', 'params': {'N': N},\n"
    "          'expected': {'TOTAL': sum(terms)}, 'terms': {'TOTAL': terms}, 'tol': 1e-6}\n",
)


def make_notebook(path, *cells):
    nb = {"cells": [{"cell_type": "code", "source": c.splitlines(keepends=True)} for c in cells]}
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith(("PARALLELISE_", "MODEL_")):
            monkeypatch.delenv(name)


def test_no_command_and_bad_arguments_exit_2(capsys):
    assert cli.main([]) == 2
    assert cli.main(["assemble", "integrate", "fortran", "x"]) == 2
    assert cli.main(["nonsense"]) == 2
    assert "usage" in capsys.readouterr().err


def test_help_and_version_exit_0(capsys):
    assert cli.main(["--help"]) == 0
    assert cli.main(["--version"]) == 0


def test_oracle_command_writes_files(tmp_path, capsys):
    nb = make_notebook(tmp_path / "harmonic.ipynb", *GOOD)
    assert cli.main(["oracle", str(nb), "--out", str(tmp_path / "o")]) == 0
    assert (tmp_path / "o" / "reference.h").exists()
    assert "harmonic" in capsys.readouterr().out


def test_oracle_command_error_is_plain(tmp_path, capsys):
    nb = make_notebook(tmp_path / "x.ipynb", "x = 1")
    assert cli.main(["oracle", str(nb)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("parallelise:") and "Add a final cell" in err
    assert "Traceback" not in err


def test_bad_configuration_exits_1_with_advice(monkeypatch, capsys):
    monkeypatch.setenv("PARALLELISE_TEMPERATURE", "9")
    assert cli.main(["health"]) == 1
    err = capsys.readouterr().err
    assert "configuration problem" in err and "PARALLELISE_TEMPERATURE" in err


def test_health_unreachable_exits_1(capsys):
    code = cli.main(["health", "--gateway-url", "http://127.0.0.1:9/v1"])
    out = capsys.readouterr()
    assert code == 1
    assert "Reachable: no" in out.out and "Could not reach" in out.err


def test_missing_work_folder_and_csv(tmp_path, capsys):
    assert cli.main(["grade", str(tmp_path / "none"), "--backend", "openmp"]) == 1
    assert "does not exist" in capsys.readouterr().err
    assert cli.main(["bench-check", str(tmp_path / "none.csv")]) == 1
    assert "does not exist" in capsys.readouterr().err


# --- the run subcommand, with stand-in assemble, check and agents --------------------------


class FakeAgent:
    name = "basic"

    def __init__(self, behaviour):
        self.behaviour = behaviour

    def run(self, work_dir, task, settings, client, step_budget=80):
        assert "make check" in task and "integrate.h" in task
        if self.behaviour == "crash":
            raise RuntimeError("boom")
        return AgentResult(
            finished=True, steps=4, requests=3, prompt_tokens=300,
            completion_tokens=60, latencies_s=[1.0, 2.0, 3.0], notes="used the notebook tool",
        )


def install_fakes(monkeypatch, tmp_path, behaviour="ok", check_pass=True):
    problems = tmp_path / "problems" / "integrate"
    problems.mkdir(parents=True)
    calls = {}

    def assemble_work_dir(problem_dir, backend, dest, include_reference=False):
        calls["assembled"] = (Path(problem_dir).name, backend, Path(dest))
        Path(dest).mkdir(parents=True)

    def run_check(work_dir, backend, timeout_s=300, speedup_floor=None):
        return SimpleNamespace(
            passed=check_pass,
            failure_stage="none" if check_pass else "benchmark",
            speedup_best=3.2 if check_pass else None,
            log="check log text",
        )

    fakes = {
        "assemble": SimpleNamespace(assemble_work_dir=assemble_work_dir),
        "check": SimpleNamespace(run_check=run_check),
        "agents": SimpleNamespace(get_agent=lambda name: FakeAgent(behaviour)),
    }
    monkeypatch.setattr(cli, "_lazy", lambda name, hint="": fakes[name])
    return calls


def run_args(tmp_path):
    return [
        "run", "integrate", "openmp", "--agent", "basic", "--model", "m1",
        "--problems-dir", str(tmp_path / "problems"),
        "--work-root", str(tmp_path / "work"),
        "--results", str(tmp_path / "runs.jsonl"),
    ]


def test_run_pass_writes_record_and_exits_0(monkeypatch, tmp_path, capsys):
    calls = install_fakes(monkeypatch, tmp_path)
    assert cli.main(run_args(tmp_path)) == 0
    assert calls["assembled"][:2] == ("integrate", "openmp")
    (record,) = ResultsStore(tmp_path / "runs.jsonl").load()
    assert record.result == "PASS" and record.failure_stage == "none"
    assert record.speedup_best == 3.2
    assert (record.requests, record.prompt_tokens, record.completion_tokens) == (3, 300, 60)
    assert record.latency_mean_s == 2.0
    assert record.tokens_per_second == 10.0  # 60 tokens over 6 seconds of waiting
    assert record.model == "m1" and not record.masked
    assert len(record.agent_version) == 10 or record.agent_version == "unknown"
    assert "used the notebook tool" in record.notes
    assert (calls["assembled"][2] / "check.log").read_text() == "check log text"
    assert "PASS" in capsys.readouterr().out


def test_run_check_failure_exits_1_and_records_stage(monkeypatch, tmp_path):
    install_fakes(monkeypatch, tmp_path, check_pass=False)
    assert cli.main(run_args(tmp_path)) == 1
    (record,) = ResultsStore(tmp_path / "runs.jsonl").load()
    assert record.result == "FAIL" and record.failure_stage == "benchmark"
    assert record.speedup_best is None


def test_run_agent_crash_is_recorded_as_agent_error(monkeypatch, tmp_path):
    install_fakes(monkeypatch, tmp_path, behaviour="crash")
    assert cli.main(run_args(tmp_path)) == 1
    (record,) = ResultsStore(tmp_path / "runs.jsonl").load()
    assert record.failure_stage == "agent-error" and "RuntimeError: boom" in record.notes


def test_run_unknown_problem(monkeypatch, tmp_path, capsys):
    install_fakes(monkeypatch, tmp_path)
    args = run_args(tmp_path)
    args[1] = "nothing"
    assert cli.main(args) == 1
    assert "no problem folder" in capsys.readouterr().err


def test_run_unavailable_agent_is_not_recorded(monkeypatch, tmp_path, capsys):
    from parallelise.errors import AgentUnavailable

    install_fakes(monkeypatch, tmp_path)

    def refuse(name):
        raise AgentUnavailable("Install with: pip install 'parallelise[agents]'")

    real = cli._lazy
    fakes_agents = SimpleNamespace(get_agent=refuse)
    monkeypatch.setattr(
        cli, "_lazy", lambda name, hint="": fakes_agents if name == "agents" else real(name)
    )
    assert cli.main(run_args(tmp_path)) == 1
    assert "pip install" in capsys.readouterr().err
    assert not (tmp_path / "runs.jsonl").exists()


def test_build_task_names_the_files():
    text = cli.build_task("scan", "cuda")
    assert "scan.cu" in text and "test_scan.c" in text and "scan.h" in text
