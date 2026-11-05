"""Tests for the command line: exit codes, messages, and the run flow with stand-in parts."""

import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from parallelise import cli
from parallelise.agents.base import AgentResult
from parallelise.model import CallRecord
from parallelise.results import ResultsStore

GOOD = (
    "N = 1000\nterms = [1.0 / (i + 1) for i in range(N)]\n",
    "ORACLE = {'params': {'N': N},\n"
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
    missing = ["check", str(tmp_path / "none"), "--problem", "integrate", "--backend", "openmp"]
    assert cli.main(missing) == 1
    assert "does not exist" in capsys.readouterr().err
    assert cli.main(["bench-check", str(tmp_path / "none.csv")]) == 1
    assert "does not exist" in capsys.readouterr().err


# --- the run subcommand, with stand-in assemble, check and agents --------------------------


class FakeAgent:
    name = "basic"
    seen: dict = {}

    def __init__(self, behaviour, routine="integrate"):
        self.behaviour = behaviour
        self.routine = routine

    def run(self, work_dir, task, settings, client, step_budget=80):
        assert "make check" in task and f"{self.routine}.h" in task
        FakeAgent.seen = {"settings": settings, "step_budget": step_budget}
        if self.behaviour == "crash":
            raise RuntimeError("boom")
        if self.behaviour == "crash-after-calls":
            client.calls.append(CallRecord(1.0, 100, 20, "ok"))
            client.calls.append(CallRecord(3.0, 100, 20, "ok"))
            raise RuntimeError("boom")
        if self.behaviour == "slow":
            time.sleep(0.15)
        result = AgentResult(
            finished=True, steps=4, requests=3, prompt_tokens=300,
            completion_tokens=60, latencies_s=[1.0, 2.0, 3.0], notes="used the notebook tool",
        )
        if self.behaviour == "time-limit":
            result.finished, result.time_limit_reached = False, True
        if self.behaviour == "error-result":
            result.finished, result.error = False, "ModelHTTPError: status 429"
        return result


def install_fakes(
    monkeypatch, tmp_path, behaviour="ok", check_pass=True, routine="integrate", check_delay=0.0,
    speed="faster", best=3.2,
):
    problems = tmp_path / "problems" / "integrate"
    problems.mkdir(parents=True)
    calls = {}

    def assemble_work_dir(problem_dir, backend, dest, include_reference=False):
        calls["assembled"] = (Path(problem_dir).name, backend, Path(dest))
        Path(dest).mkdir(parents=True)

    def require_reference_test(problem_dir, backend):
        calls["required"] = (Path(problem_dir), backend)

    def run_check(work_dir, problem_dir, backend, timeout_s=300, speedup_floor=None,
                  as_assembled=False):
        calls["checked"] = (Path(work_dir), Path(problem_dir), backend)
        calls["as_assembled"] = as_assembled
        time.sleep(check_delay)
        return SimpleNamespace(
            passed=check_pass,
            failure_stage="none" if check_pass else "benchmark",
            speedup_best=best if check_pass else None,
            log="check log text",
            speed_verdict=speed if check_pass else "not measured",
            speedup_floor=1.25,
        )

    fakes = {
        "assemble": SimpleNamespace(assemble_work_dir=assemble_work_dir),
        "check": SimpleNamespace(
            run_check=run_check, require_reference_test=require_reference_test
        ),
        "agents": SimpleNamespace(get_agent=lambda name: FakeAgent(behaviour, routine)),
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
    assert record.speed_verdict == "faster"
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "PASS: correct; faster on this machine (best 3.20, floor 1.25)"
    # The check is given the problem folder the work folder was built from.
    assert calls["checked"][1] == tmp_path / "problems" / "integrate"
    assert calls["checked"][0] == calls["assembled"][2] and calls["checked"][2] == "openmp"
    assert calls["required"] == (tmp_path / "problems" / "integrate", "openmp")


def test_run_check_failure_exits_1_and_records_stage(monkeypatch, tmp_path, capsys):
    install_fakes(monkeypatch, tmp_path, check_pass=False)
    assert cli.main(run_args(tmp_path)) == 1
    (record,) = ResultsStore(tmp_path / "runs.jsonl").load()
    assert record.result == "FAIL" and record.failure_stage == "benchmark"
    assert record.speedup_best is None
    assert record.speed_verdict == "not measured"
    assert capsys.readouterr().out.splitlines()[-1] == (
        "FAIL: the benchmark did not run, gave a wrong answer at its own size, or its table "
        "was malformed, invented or not borne out by the timed calls (stage benchmark); "
        "speed not measured"
    )


def test_run_of_a_correct_routine_that_is_not_faster_exits_0(monkeypatch, tmp_path, capsys):
    install_fakes(monkeypatch, tmp_path, speed="no useful speed-up", best=1.07)
    assert cli.main(run_args(tmp_path)) == 0
    (record,) = ResultsStore(tmp_path / "runs.jsonl").load()
    assert (record.result, record.failure_stage) == ("PASS", "none")
    assert (record.speed_verdict, record.speedup_best) == ("no useful speed-up", 1.07)
    assert capsys.readouterr().out.splitlines()[-1] == (
        "PASS: correct; no useful speed-up on this machine (best 1.07, floor 1.25)"
    )


@pytest.mark.parametrize(
    ("args", "line"),
    [
        ((True, "none", "faster", 3.456, 1.5), "PASS: correct; faster on this machine "
                                               "(best 3.46, floor 1.5)"),
        ((True, "none", "no useful speed-up", 1.07, None),
         "PASS: correct; no useful speed-up on this machine (best 1.07)"),
        ((True, "none", "speed not judged on this machine", 1.07, 1.3),
         "PASS: correct; speed not judged on this machine (best 1.07)"),
        ((True, "none", None, None, None), "PASS: correct; speed not measured"),
        ((False, "test-many-workers", "not measured", None, 1.3),
         "FAIL: the problem's test failed with several workers (stage test-many-workers); "
         "speed not measured"),
        ((False, "tampered", "not measured", None, None),
         "FAIL: a trusted file in the work folder was changed (stage tampered); "
         "speed not measured"),
    ],
    ids=["faster", "no-speed-up", "not-judged", "no-verdict", "wrong-answer", "tampered"],
)
def test_verdict_line_gives_correctness_and_speed_together(args, line):
    assert cli.verdict_line(*args) == line


def test_masked_run_is_checked_against_the_masked_problem(monkeypatch, tmp_path):
    calls = install_fakes(monkeypatch, tmp_path, routine="problem_a")
    fakes = {}

    def mask_problem(problem_dir, dest, alias):
        fakes["masked"] = (Path(problem_dir), Path(dest), alias)
        Path(dest).mkdir(parents=True)

    real = cli._lazy
    monkeypatch.setattr(
        cli, "_lazy",
        lambda name, hint="": SimpleNamespace(mask_problem=mask_problem)
        if name == "study.mask" else real(name),
    )
    assert cli.main([*run_args(tmp_path), "--masked", "problem_a"]) == 0
    masked_dir = tmp_path / "work" / "_masked" / "integrate"
    assert fakes["masked"][1] == masked_dir
    assert calls["assembled"][0] == "integrate"
    assert calls["checked"][1] == masked_dir
    assert calls["required"][0] == masked_dir


def test_run_agent_crash_is_still_checked_and_both_are_recorded(monkeypatch, tmp_path):
    calls = install_fakes(monkeypatch, tmp_path, behaviour="crash", check_pass=False)
    assert cli.main(run_args(tmp_path)) == 1
    (record,) = ResultsStore(tmp_path / "runs.jsonl").load()
    assert "checked" in calls, "an agent crash must not skip the check"
    assert record.result == "FAIL" and record.failure_stage == "benchmark"  # the check's stage
    assert "RuntimeError: boom" in record.notes
    assert (calls["assembled"][2] / "check.log").read_text() == "check log text"


def test_run_whatever_is_on_disk_after_a_crash_can_still_pass(monkeypatch, tmp_path):
    install_fakes(monkeypatch, tmp_path, behaviour="crash", check_pass=True)
    assert cli.main(run_args(tmp_path)) == 0
    (record,) = ResultsStore(tmp_path / "runs.jsonl").load()
    assert record.result == "PASS" and record.failure_stage == "none"
    assert "RuntimeError: boom" in record.notes


def test_run_error_reported_in_the_result_is_noted_and_checked(monkeypatch, tmp_path):
    calls = install_fakes(monkeypatch, tmp_path, behaviour="error-result", check_pass=False)
    assert cli.main(run_args(tmp_path)) == 1
    (record,) = ResultsStore(tmp_path / "runs.jsonl").load()
    assert "checked" in calls and record.failure_stage == "benchmark"
    assert "ModelHTTPError: status 429" in record.notes
    assert "step budget" not in record.notes  # the error, not the budget, ended the run
    assert (record.requests, record.prompt_tokens) == (3, 300)  # what the agent counted is kept


def test_run_keeps_the_usage_the_client_recorded_when_the_agent_crashes(monkeypatch, tmp_path):
    install_fakes(monkeypatch, tmp_path, behaviour="crash-after-calls")
    cli.main(run_args(tmp_path))
    (record,) = ResultsStore(tmp_path / "runs.jsonl").load()
    assert (record.requests, record.prompt_tokens, record.completion_tokens) == (2, 200, 40)
    assert record.latency_mean_s == 2.0 and record.tokens_per_second == 10.0


def test_run_time_limit_stop_is_noted_and_the_check_decides_the_stage(monkeypatch, tmp_path):
    install_fakes(monkeypatch, tmp_path, behaviour="time-limit", check_pass=False)
    assert cli.main([*run_args(tmp_path), "--time-limit", "90"]) == 1
    (record,) = ResultsStore(tmp_path / "runs.jsonl").load()
    assert record.failure_stage == "benchmark"  # the check's stage, not a stage of its own
    assert "The time limit of 90 seconds stopped the agent." in record.notes
    assert record.time_limit_s == 90
    assert "step budget" not in record.notes


def test_run_time_limit_comes_from_the_option_then_the_environment(monkeypatch, tmp_path):
    install_fakes(monkeypatch, tmp_path)
    cli.main(run_args(tmp_path))
    assert FakeAgent.seen["settings"].time_limit_s == 1800  # the default
    monkeypatch.setenv("PARALLELISE_TIME_LIMIT", "700")
    cli.main([*run_args(tmp_path), "--force"])
    assert FakeAgent.seen["settings"].time_limit_s == 700
    cli.main([*run_args(tmp_path), "--force", "--time-limit", "45"])
    assert FakeAgent.seen["settings"].time_limit_s == 45


def test_run_records_the_agent_time_and_the_check_time_separately(monkeypatch, tmp_path):
    install_fakes(monkeypatch, tmp_path, behaviour="slow", check_delay=0.3)
    cli.main(run_args(tmp_path))
    (record,) = ResultsStore(tmp_path / "runs.jsonl").load()
    assert 0.15 <= record.agent_time_s < 0.3
    assert 0.3 <= record.check_time_s < 0.6
    assert record.wall_time_s >= record.agent_time_s + record.check_time_s - 0.02


def test_run_record_carries_the_settings_and_the_environment(monkeypatch, tmp_path):
    import parallelise

    install_fakes(monkeypatch, tmp_path)
    monkeypatch.setenv("PARALLELISE_TEMPERATURE", "0.5")
    cli.main([*run_args(tmp_path), "--step-budget", "7"])
    (record,) = ResultsStore(tmp_path / "runs.jsonl").load()
    assert record.temperature == 0.5
    assert record.step_budget == 7 and FakeAgent.seen["step_budget"] == 7
    assert record.time_limit_s == 1800
    assert record.package_version == parallelise.__version__
    assert re.fullmatch(r"[0-9a-f]{12}", record.source_hash)
    assert record.cpu_count and record.cpu_count >= 1
    assert re.fullmatch(r"\d+\.\d+.*", record.framework_versions["httpx"])
    assert set(record.framework_versions) <= {
        "httpx", "httpx2", "pydantic-ai", "langgraph", "langchain", "langchain-openai", "openai"}


def test_run_step_budget_must_be_a_whole_number_of_one_or_more(monkeypatch, tmp_path, capsys):
    install_fakes(monkeypatch, tmp_path)
    assert cli.main([*run_args(tmp_path), "--step-budget", "0"]) == 1
    assert "step budget" in capsys.readouterr().err


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


# --- the shape of the command list and the old names ---------------------------------------


def test_main_help_lists_only_the_five_commands(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for name in ("health", "assess", "run", "check", "study"):
        assert f"\n    {name} " in out
    for hidden in ("grade", "mask", "report", "oracle", "assemble", "bench-check"):
        assert f"\n    {hidden} " not in out


def test_every_command_help_has_an_example(capsys):
    for command in (["health"], ["assess"], ["run"], ["check"], ["study", "mask"],
                    ["study", "report"], ["oracle"], ["assemble"], ["bench-check"]):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([*command, "--help"])
        assert "Example: parallelise" in capsys.readouterr().out


def test_grade_is_a_hidden_alias_of_check(tmp_path, capsys):
    arguments = [str(tmp_path / "none"), "--problem", "integrate", "--backend", "openmp"]
    assert cli.main(["grade", *arguments]) == 1
    err = capsys.readouterr().err
    assert "now called 'check'" in err and "does not exist" in err


def test_check_needs_the_problem_as_well_as_the_work_folder(tmp_path, capsys):
    assert cli.main(["check", str(tmp_path), "--backend", "openmp"]) == 2
    assert "--problem" in capsys.readouterr().err


def test_check_passes_the_resolved_problem_folder_to_the_check(monkeypatch, tmp_path, capsys):
    calls = install_fakes(monkeypatch, tmp_path, check_pass=False)
    work = tmp_path / "finished"
    work.mkdir()
    by_name = ["check", str(work), "--problem", "integrate", "--backend", "openmp",
               "--problems-dir", str(tmp_path / "problems")]  # fmt: skip
    assert cli.main(by_name) == 1  # the stand-in check fails
    assert calls["checked"] == (work, tmp_path / "problems" / "integrate", "openmp")
    assert capsys.readouterr().out.splitlines()[-1].startswith("FAIL: the benchmark did not run")
    by_path = ["check", str(work), "--problem", str(tmp_path / "problems" / "integrate"),
               "--backend", "openmp"]  # fmt: skip
    assert cli.main(by_path) == 1
    assert calls["checked"][1] == tmp_path / "problems" / "integrate"


@pytest.mark.parametrize(
    ("speed", "best", "line"),
    [
        ("faster", 3.2, "PASS: correct; faster on this machine (best 3.20, floor 1.25)"),
        ("no useful speed-up", 1.07,
         "PASS: correct; no useful speed-up on this machine (best 1.07, floor 1.25)"),
    ],
)
def test_check_of_a_correct_routine_exits_0_whatever_its_speed(
    monkeypatch, tmp_path, capsys, speed, best, line
):
    install_fakes(monkeypatch, tmp_path, speed=speed, best=best)
    work = tmp_path / "finished"
    work.mkdir()
    args = ["check", str(work), "--problem", "integrate", "--backend", "openmp",
            "--problems-dir", str(tmp_path / "problems")]  # fmt: skip
    assert cli.main(args) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "check log text" and out[-1] == line


def test_check_with_an_unknown_problem_says_what_to_do(monkeypatch, tmp_path, capsys):
    install_fakes(monkeypatch, tmp_path)
    work = tmp_path / "finished"
    work.mkdir()
    args = ["check", str(work), "--problem", "nothing", "--backend", "openmp",
            "--problems-dir", str(tmp_path / "problems")]  # fmt: skip
    assert cli.main(args) == 1
    assert "no problem folder" in capsys.readouterr().err


def test_check_help_explains_the_fresh_folder(capsys):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["check", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "rebuilds a fresh folder from the problem's own files" in text
    assert "only the agent's routine and the problem's own known-good test" in text
    assert "--problem" in text


def test_mask_and_report_old_names_print_a_notice(tmp_path, capsys):
    results = str(tmp_path / "none.jsonl")
    assert cli.main(["report", "--results", results]) == 1
    assert "parallelise study report" in capsys.readouterr().err
    assert cli.main(["study", "report", "--results", results]) == 1
    assert "There are no run records" in capsys.readouterr().err
    assert cli.main(["mask", str(tmp_path / "x"), "problem_a", str(tmp_path / "y")]) == 1
    assert "parallelise study mask" in capsys.readouterr().err


def test_agent_version_changes_when_a_prompt_changes(monkeypatch, tmp_path):
    """A change of wording in the shared prompts must give a new agent version."""
    import shutil as _shutil
    from pathlib import Path as _Path

    from parallelise import cli as cli_module
    from parallelise.agents import get_agent

    agent = get_agent("basic")
    before = cli_module._agent_version(agent)
    assert before != "unknown" and len(before) == 10

    package = _Path(cli_module.__file__).resolve().parent
    copy = tmp_path / "parallelise"
    _shutil.copytree(package, copy)
    (copy / "agents" / "base.py").write_text(
        (package / "agents" / "base.py").read_text() + "\n# a changed prompt\n"
    )
    monkeypatch.setattr(cli_module, "__file__", str(copy / "cli.py"))
    after = cli_module._agent_version(agent)
    assert after != before
