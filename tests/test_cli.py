"""Tests for the command line: exit statuses, messages, and the run with stand-in parts.

The run tests use a copy of the midpoint fixture as the problem, a stand-in agent that writes
the routine file, and a stand-in judgement, so that they need neither a model nor a compiler.
"""

import os
import re
import shutil
from pathlib import Path

import pytest

from parallelise import cli
from parallelise.agents.base import AgentResult
from parallelise.errors import AgentUnavailable
from parallelise.judge import Verdict
from parallelise.model import Call
from parallelise.results import ResultsStore

FIXTURE = Path(__file__).parent / "fixtures" / "midpoint"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("PARALLELISE_"):
            monkeypatch.delenv(name)


def test_wrong_arguments_exit_2_and_help_and_version_exit_0(capsys):
    assert cli.main([]) == 2
    assert cli.main(["run", "midpoint", "--backend", "fortran"]) == 2
    assert cli.main(["--help"]) == 0 and cli.main(["--version"]) == 0


def test_every_command_is_listed(capsys):
    cli.main(["--help"])
    listing = capsys.readouterr().out
    assert all(command in listing for command in ("serial", "run", "check", "reverse", "report"))


# --- run, with a stand-in agent and judgement ----------------------------------------------------


class FakeAgent:
    seen: dict = {}

    def __init__(self, behaviour):
        self.behaviour = behaviour

    def run(self, work_dir, task, settings, client, step_budget, toolset, system_prompt):
        FakeAgent.seen = {"settings": settings, "task": task, "prompt": system_prompt}
        client.meter.calls += [Call(1.0, 100, 20), Call(3.0, 200, 40)]
        if self.behaviour == "crash":
            raise RuntimeError("boom")
        name = Path(work_dir).joinpath("serial.c").read_text().split("double ")[1].split("(")[0]
        (Path(work_dir) / f"{name}.c").write_text("/* the routine */\n")
        if self.behaviour == "time-limit":
            return AgentResult(False, time_limit_reached=True)
        if self.behaviour == "error":
            return AgentResult(False, error="ModelError: the service refused")
        return AgentResult(True, "the checks passed")


def install_fakes(monkeypatch, tmp_path, behaviour="ok", verdict=None):
    problem = tmp_path / "midpoint"
    problem.mkdir()
    shutil.copy(FIXTURE / "midpoint.ipynb", problem)
    shutil.copy(FIXTURE / "serial.c", problem)
    calls: dict = {}

    def judge(nb, decl, routine, serial, backend, timeout_s=900.0, name=None):
        calls["judged"] = (Path(routine), Path(serial), backend, name)
        if not Path(routine).is_file():  # as the judgement itself says
            return Verdict(False, "no-files")
        return verdict or Verdict(True, "none", 3.2, True, 2.0, True, log=["the judgement's log"])

    monkeypatch.setattr(cli.judge, "judge", judge)
    monkeypatch.setattr(cli.judge, "missing_tools", lambda backend: [])
    monkeypatch.setattr(cli, "get_agent", lambda name: FakeAgent(behaviour))
    return calls


def run(tmp_path, *extra):
    return cli.main(["run", str(tmp_path / "midpoint"), "--backend", "openmp", "--model", "m1",
                     "--work-root", str(tmp_path / "work"),
                     "--results", str(tmp_path / "runs.jsonl"), *extra])  # fmt: skip


def records(tmp_path):
    return ResultsStore(tmp_path / "runs.jsonl").load()


def work_folder(tmp_path):
    return sorted((tmp_path / "work" / "midpoint" / "openmp" / "m1" / "basic").iterdir())[-1]


def test_a_correct_run_is_recorded_with_the_meter_s_counts_and_exits_0(monkeypatch, tmp_path,
                                                                       capsys):
    calls = install_fakes(monkeypatch, tmp_path)
    assert run(tmp_path, "--repeat", "2", "--step-budget", "7", "--serving", "provider p",
               "--quantisation", "fp8") == 0
    (record,) = records(tmp_path)
    assert (record.correct, record.failure_stage, record.speedup, record.faster) == (
        True, "none", 3.2, True)
    assert (record.requests, record.prompt_tokens, record.completion_tokens) == (2, 300, 60)
    assert record.latency_mean_s == 2.0 and record.tokens_per_second == 15.0
    assert (record.repeat, record.step_budget, record.model) == (2, 7, "m1")
    assert (record.serving, record.quantisation) == ("provider p", "fp8")
    assert re.fullmatch(r"[0-9a-f]{12}", record.source_hash)
    assert re.fullmatch(r"[0-9a-f]{12}", record.serial_hash)
    work = work_folder(tmp_path)
    assert record.work_folder == str(work)
    assert (work / "check.log").read_text() == "the judgement's log\n"
    assert calls["judged"] == (work / "midpoint.c", work / "serial.c", "openmp", "midpoint")
    assert capsys.readouterr().out.splitlines()[-1] == (
        "PASS: correct; faster than the serial routine (speed-up 3.20)")


def test_the_agent_is_told_nothing_that_names_an_expected_value(monkeypatch, tmp_path):
    install_fakes(monkeypatch, tmp_path)
    run(tmp_path)
    text = FakeAgent.seen["task"] + FakeAgent.seen["prompt"]
    assert "expected" not in text and "reference" not in text and "serial.c" in text


def test_a_wrong_routine_is_recorded_with_its_stage_and_exits_1(monkeypatch, tmp_path):
    install_fakes(monkeypatch, tmp_path, verdict=Verdict(False, "benchmark"))
    assert run(tmp_path) == 1
    (record,) = records(tmp_path)
    assert not record.correct and record.failure_stage == "benchmark" and record.speedup is None


def test_a_masked_run_hides_the_name_and_is_judged_under_the_alias(monkeypatch, tmp_path):
    calls = install_fakes(monkeypatch, tmp_path)
    assert run(tmp_path, "--masked", "problem_a") == 0
    work = work_folder(tmp_path)
    for name in ("function.py", "serial.c", "problem_a.h"):
        assert "midpoint" not in (work / name).read_text().lower()
    assert calls["judged"][3] == "problem_a" and records(tmp_path)[0].masked


@pytest.mark.parametrize(("behaviour", "words", "stage"), [
    ("crash", "RuntimeError: boom", "no-files"),
    ("error", "The agent stopped with ModelError: the service refused.", "none"),
    ("time-limit", "The time limit of 1800 seconds stopped the agent.", "none"),
])  # fmt: skip
def test_whatever_becomes_of_the_agent_the_routine_is_judged_and_noted(monkeypatch, tmp_path,
                                                                        behaviour, words, stage):
    install_fakes(monkeypatch, tmp_path, behaviour)
    run(tmp_path)
    (record,) = records(tmp_path)
    assert words in record.notes and record.failure_stage == stage and record.requests == 2


def test_run_says_how_to_make_a_missing_serial_routine(monkeypatch, tmp_path, capsys):
    install_fakes(monkeypatch, tmp_path)
    (tmp_path / "midpoint" / "serial.c").unlink()
    assert run(tmp_path) == 1
    assert "parallelise serial" in capsys.readouterr().err and not records(tmp_path)


@pytest.mark.parametrize("where", ["results", "work-root"])
def test_a_place_that_cannot_be_written_is_found_before_the_agent_starts(monkeypatch, tmp_path,
                                                                          capsys, where):
    install_fakes(monkeypatch, tmp_path)
    FakeAgent.seen = {}
    (tmp_path / "blocker").write_text("a file, not a folder")
    assert run(tmp_path, f"--{where}", str(tmp_path / "blocker" / "inside")) == 1
    assert "blocker" in capsys.readouterr().err and FakeAgent.seen == {}


def test_an_agent_that_is_not_installed_makes_no_record(monkeypatch, tmp_path, capsys):
    install_fakes(monkeypatch, tmp_path)

    def refuse(name):
        raise AgentUnavailable("install with: pip install 'parallelise[agents]'")

    monkeypatch.setattr(cli, "get_agent", refuse)
    assert run(tmp_path) == 1
    assert "pip install" in capsys.readouterr().err and not records(tmp_path)


def test_a_folder_without_a_notebook_of_its_name_is_not_a_problem(tmp_path, capsys):
    assert cli.main(["run", str(tmp_path), "--backend", "openmp"]) == 1
    assert "not a problem folder" in capsys.readouterr().err


@pytest.mark.parametrize(("verdict", "line"), [
    (Verdict(True, "none", 3.456, True), "PASS: correct; faster than the serial routine "
                                         "(speed-up 3.46)"),
    (Verdict(True, "none", 0.9, False), "PASS: correct; not faster than the serial routine "
                                        "(speed-up 0.90)"),
    (Verdict(False, "rule"), "FAIL: the routine breaks the rule on what a routine may contain "
                             "(stage rule); speed not measured"),
    (Verdict(False, "notebook"), "NOT JUDGED: the notebook's own Python failed during the "
                                 "judgement (stage notebook)"),
])  # fmt: skip
def test_the_closing_line_gives_correctness_and_speed_together(verdict, line):
    assert cli.verdict_line(verdict) == line


# --- check and serial --------------------------------------------------------------------------


def test_check_judges_a_finished_work_folder_against_its_problem(monkeypatch, tmp_path, capsys):
    calls = install_fakes(monkeypatch, tmp_path)
    run(tmp_path)
    work = work_folder(tmp_path)
    assert cli.main(["check", str(work), "--problem", str(tmp_path / "midpoint"),
                     "--backend", "openmp"]) == 0  # fmt: skip
    assert calls["judged"][0] == work / "midpoint.c"
    assert capsys.readouterr().out.splitlines()[-1].startswith("PASS")


def test_check_refuses_a_folder_that_run_did_not_make(monkeypatch, tmp_path, capsys):
    install_fakes(monkeypatch, tmp_path)
    assert cli.main(["check", str(tmp_path / "midpoint"), "--problem", str(tmp_path / "midpoint"),
                     "--backend", "openmp"]) == 1  # fmt: skip
    assert "not a work folder" in capsys.readouterr().err


def test_serial_will_not_replace_a_serial_routine_without_force(monkeypatch, tmp_path, capsys):
    install_fakes(monkeypatch, tmp_path)
    assert cli.main(["serial", str(tmp_path / "midpoint")]) == 1
    assert "--force" in capsys.readouterr().err


def test_an_unexpected_file_error_is_one_plain_line(monkeypatch, capsys):
    def fail(args):
        raise PermissionError(13, "Permission denied", "/some/file")

    monkeypatch.setattr(cli, "cmd_report", fail)
    assert cli.main(["report"]) == 1
    assert capsys.readouterr().err.strip() == (
        "parallelise: Permission denied: /some/file. Check the path and its permissions.")
