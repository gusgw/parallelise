"""Ordinary misuse must end in one plain line, never in a traceback.

Every case below is something a user can do by mistake: give a name the tool does not accept,
point it at a file that is unreadable or unwritable, or hand it a malformed data file. The
command line must print one line that begins "parallelise:" and says what to do, and return
status 1.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest
from tests.test_cli import GOOD, install_fakes, make_notebook, run_args

from parallelise import cli
from parallelise.agents import get_agent
from parallelise.errors import CheckError, ParalleliseError

FIXTURE = Path(__file__).parent / "fixtures" / "midpoint"
NOT_ROOT = pytest.mark.skipif(os.geteuid() == 0, reason="the root user may write anywhere")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("PARALLELISE_", "MODEL_")):
            monkeypatch.delenv(name)


def fails_plainly(argv, capsys, *words):
    """Run the command line and require status 1 with one plain line on standard error."""
    code = cli.main([str(a) for a in argv])
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if line.strip()]
    assert code == 1, err
    assert len(lines) == 1 and lines[0].startswith("parallelise:"), err
    assert "Traceback" not in err and "File \"" not in err, err
    for word in words:
        assert word in lines[0], lines[0]


def notebook(path: Path, text: str | bytes) -> Path:
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text)
    return path


def good_store(path: Path, **extra) -> Path:
    record = {"problem": "a", "backend": "openmp", "model": "m", "agent": "basic",
              "result": "PASS", **extra}
    path.write_text(json.dumps(record) + "\n")
    return path


# --- masking -------------------------------------------------------------------------------


def test_mask_alias_in_the_word_list(tmp_path, capsys):
    fails_plainly(["study", "mask", FIXTURE, "jacobi", tmp_path / "out"], capsys, "jacobi")


def test_mask_alias_that_is_not_an_identifier(tmp_path, capsys):
    fails_plainly(["study", "mask", FIXTURE, "bad name", tmp_path / "out"], capsys, "bad name")


def test_mask_problem_without_a_notebook(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    fails_plainly(["study", "mask", empty, "problem_a", tmp_path / "out"], capsys, "notebook")


def test_mask_destination_that_is_a_file(tmp_path, capsys):
    target = notebook(tmp_path / "afile", "x")
    fails_plainly(["study", "mask", FIXTURE, "problem_a", target], capsys, "afile", "not a folder")


@NOT_ROOT
def test_mask_destination_that_cannot_be_written(tmp_path, capsys):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        fails_plainly(["study", "mask", FIXTURE, "problem_a", locked / "out"], capsys, "locked")
    finally:
        locked.chmod(0o700)


def test_mask_notebook_that_is_not_json(tmp_path, capsys):
    problem = tmp_path / "broken"
    problem.mkdir()
    notebook(problem / "broken.ipynb", "this is not json")
    fails_plainly(["study", "mask", problem, "problem_a", tmp_path / "out"], capsys, "broken.ipynb")


def test_old_mask_name_fails_plainly_too(tmp_path, capsys):
    code = cli.main(["mask", str(FIXTURE), "jacobi", str(tmp_path / "out")])
    err = capsys.readouterr().err
    assert code == 1 and "Traceback" not in err and "jacobi" in err


# --- the oracle command ----------------------------------------------------------------------


def test_oracle_notebook_that_is_not_json(tmp_path, capsys):
    path = notebook(tmp_path / "bad.ipynb", "this is not json")
    fails_plainly(["oracle", path], capsys, "bad.ipynb", "not a Jupyter notebook")


def test_oracle_notebook_that_is_not_text(tmp_path, capsys):
    path = notebook(tmp_path / "bad.ipynb", b"\xff\xfe\x00")
    fails_plainly(["oracle", path], capsys, "bad.ipynb", "not a Jupyter notebook")


def test_oracle_output_folder_that_is_a_file(tmp_path, capsys):
    target = notebook(tmp_path / "afile", "x")
    harmonic = make_notebook(tmp_path / "harmonic.ipynb", *GOOD)
    fails_plainly(["oracle", harmonic, "--out", target], capsys, "afile")


@NOT_ROOT
def test_oracle_output_folder_that_cannot_be_written(tmp_path, capsys):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    harmonic = make_notebook(tmp_path / "harmonic.ipynb", *GOOD)
    try:
        fails_plainly(["oracle", harmonic, "--out", locked / "o"], capsys, "locked")
    finally:
        locked.chmod(0o700)


# --- assembling ------------------------------------------------------------------------------


def test_assemble_destination_that_is_a_file(tmp_path, capsys):
    target = notebook(tmp_path / "afile", "x")
    fails_plainly(["assemble", FIXTURE, "openmp", target], capsys, "afile", "not a folder")


@NOT_ROOT
def test_assemble_destination_that_cannot_be_written(tmp_path, capsys):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        fails_plainly(["assemble", FIXTURE, "openmp", locked / "w"], capsys, "locked")
    finally:
        locked.chmod(0o700)


# --- the results store and the report ----------------------------------------------------------


def test_report_store_that_is_not_text(tmp_path, capsys):
    store = notebook(tmp_path / "runs.jsonl", b"\xff\xfe\x00\n")
    fails_plainly(["study", "report", "--results", store], capsys, "runs.jsonl")


def test_report_store_that_is_a_folder(tmp_path, capsys):
    store = tmp_path / "runs.jsonl"
    store.mkdir()
    fails_plainly(["study", "report", "--results", store], capsys, "runs.jsonl")


def test_report_record_with_a_bad_date(tmp_path, capsys):
    store = good_store(tmp_path / "runs.jsonl", when="yesterday")
    fails_plainly(["study", "report", "--results", store], capsys, "yesterday")


def test_report_record_with_a_field_that_is_not_a_number(tmp_path, capsys):
    store = good_store(tmp_path / "runs.jsonl", requests="lots")
    fails_plainly(["study", "report", "--results", store], capsys, "requests", "lots")


@pytest.mark.parametrize(
    "text, words",
    [
        (None, ("prices.json", "cannot be read")),
        ("{oops", ("prices.json", "not valid JSON")),
        ("[1, 2]", ("prices.json", "object")),
        (json.dumps({"m": {"input": 1}}), ("prices.json", "output")),
        (json.dumps({"m": {"input": "x", "output": 2}}), ("prices.json", "number")),
    ],
)
def test_report_malformed_price_file(tmp_path, capsys, text, words):
    store = good_store(tmp_path / "runs.jsonl")
    prices = tmp_path / "prices.json"
    if text is not None:
        prices.write_text(text)
    fails_plainly(["study", "report", "--results", store, "--prices", prices], capsys, *words)


@NOT_ROOT
def test_report_chart_that_cannot_be_written(tmp_path, capsys):
    store = good_store(tmp_path / "runs.jsonl")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        fails_plainly(
            ["study", "report", "--results", store, "--chart", locked / "c.svg"], capsys, "locked"
        )
    finally:
        locked.chmod(0o700)


def test_bench_check_file_that_is_not_text(tmp_path, capsys):
    path = notebook(tmp_path / "bench.csv", b"\xff\xfe\x00")
    code = cli.main(["bench-check", str(path)])
    captured = capsys.readouterr()
    assert code == 1 and "Traceback" not in captured.err
    assert "rejected" in captured.out


# --- settings --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url", ["https://host:notaport/v1", "http://[::1/v1", "http://127.0.0.1:99999/v1"]
)
def test_gateway_address_that_is_not_a_web_address(capsys, url):
    fails_plainly(["health", "--gateway-url", url], capsys, "PARALLELISE_GATEWAY_URL")


def test_api_key_that_cannot_be_sent(monkeypatch, capsys):
    monkeypatch.setenv("PARALLELISE_API_KEY", "kéy")
    fails_plainly(["health"], capsys, "PARALLELISE_API_KEY")


@pytest.mark.parametrize(
    "variable, value",
    [
        ("PARALLELISE_TIME_LIMIT", "0"),
        ("PARALLELISE_TIME_LIMIT", "soon"),
        ("PARALLELISE_THROTTLE_WAIT", "-1"),
        ("PARALLELISE_THROTTLE_WAIT", "long"),
    ],
)
def test_bad_time_settings_fail_plainly(monkeypatch, capsys, variable, value):
    monkeypatch.setenv(variable, value)
    fails_plainly(["health"], capsys, variable)


# --- the run command ---------------------------------------------------------------------------


def test_run_results_file_that_cannot_be_written_is_found_before_the_agent_starts(
    monkeypatch, tmp_path, capsys
):
    calls = install_fakes(monkeypatch, tmp_path)
    blocker = notebook(tmp_path / "afile", "x")
    args = run_args(tmp_path)
    args[args.index("--results") + 1] = str(blocker / "runs.jsonl")
    fails_plainly(args, capsys, "afile", "results")
    assert "assembled" not in calls, "the agent's work folder was made before the check"


def test_run_results_path_that_is_a_folder(monkeypatch, tmp_path, capsys):
    install_fakes(monkeypatch, tmp_path)
    args = run_args(tmp_path)
    args[args.index("--results") + 1] = str(tmp_path)
    fails_plainly(args, capsys, "results")


def test_run_work_root_that_is_a_file(monkeypatch, tmp_path, capsys):
    install_fakes(monkeypatch, tmp_path)
    blocker = notebook(tmp_path / "afile", "x")
    args = run_args(tmp_path)
    args[args.index("--work-root") + 1] = str(blocker)
    fails_plainly(args, capsys, "afile")


def test_run_masked_alias_in_the_word_list(monkeypatch, tmp_path, capsys):
    from parallelise.study import mask

    install_fakes(monkeypatch, tmp_path)
    fakes = cli._lazy
    monkeypatch.setattr(
        cli, "_lazy", lambda name, hint="": mask if name == "study.mask" else fakes(name)
    )
    fails_plainly([*run_args(tmp_path), "--masked", "jacobi"], capsys, "jacobi")


@pytest.mark.parametrize(
    "option, value, word",
    [("--step-budget", "0", "step budget"), ("--check-timeout", "0", "check timeout"),
     ("--time-limit", "0", "time limit")],
)
def test_run_numbers_that_cannot_work(monkeypatch, tmp_path, capsys, option, value, word):
    install_fakes(monkeypatch, tmp_path)
    fails_plainly([*run_args(tmp_path), option, value], capsys, word)


# --- names and the check itself --------------------------------------------------------------


def test_unknown_agent_name_is_a_plain_error():
    with pytest.raises(ParalleliseError, match="nonsense"):
        get_agent("nonsense")


def test_an_unexpected_file_error_is_still_one_plain_line(monkeypatch, capsys):
    def refuse(args):
        raise PermissionError(13, "Permission denied", "/somewhere/runs.jsonl")

    monkeypatch.setattr(cli, "cmd_health", refuse)
    fails_plainly(["health"], capsys, "Permission denied", "/somewhere/runs.jsonl")


def test_the_check_reports_that_it_could_not_run(monkeypatch, tmp_path):
    """When the check cannot even start, CheckError says so; nothing else raises it."""
    from parallelise.check import run_check

    work = tmp_path / "work"
    work.mkdir()
    missing = tmp_path / "no-such-temporary-folder"
    monkeypatch.setattr(tempfile, "tempdir", str(missing))
    with pytest.raises(CheckError, match="could not run"):
        run_check(work, FIXTURE, "openmp")
