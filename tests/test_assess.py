"""Tests for the backend-assessment stage, against the scripted model server."""

import json
from pathlib import Path

import pytest
from tests.fake_model_server import FakeModelServer, finish_step
from tests.test_cli import install_fakes, make_notebook

from parallelise import assess, cli
from parallelise.config import Settings
from parallelise.errors import ParalleliseError
from parallelise.model import ModelClient
from parallelise.results import ResultsStore

VERDICT = {
    "backend": "openmp",
    "parts": [],
    "dependency_structure": "a sum of independent terms",
    "bound_by": "compute",
    "communication": "one number per process at the end",
    "also_suitable": ["mpi"],
    "poor_fit": ["cuda"],
    "expected_speedup": "close to the number of cores",
    "justification": "Each term is independent and the sum is a reduction.",
}


def reply(verdict: dict, style: str = "plain") -> str:
    text = json.dumps(verdict)
    analysis = "The loop is a reduction over independent terms.\n\n"
    if style == "fence":
        return f"{analysis}```json\n{text}\n```\nThat is my view."
    if style == "prose":
        return f"{analysis}My verdict follows, and I stand by it: {text} Thank you."
    return analysis + text


def served(text: str) -> FakeModelServer:
    return FakeModelServer([finish_step(text, 500, 120)])


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith(("PARALLELISE_", "MODEL_")):
            monkeypatch.delenv(name)


@pytest.fixture
def notebook(tmp_path):
    return make_notebook(tmp_path / "integrate.ipynb", "s = 0\nfor i in range(10): s += i\n")


def assess_with(server: FakeModelServer, notebook: Path):
    settings = Settings.from_env({"gateway_url": server.url, "model": "m1"})
    with ModelClient(settings) as client:
        return assess.assess_notebook(notebook, client), client


@pytest.mark.parametrize("style", ["plain", "fence", "prose"])
def test_assess_parses_plain_fenced_and_wrapped_replies(notebook, style):
    with served(reply(VERDICT, style)) as server:
        verdict, client = assess_with(server, notebook)
    assert verdict.backend == "openmp" and verdict.chosen_backend == "openmp"
    assert verdict.also_suitable == ["mpi"] and verdict.poor_fit == ["cuda"]
    assert verdict.model == "m1" and verdict.notebook == "integrate.ipynb"
    assert "reduction over independent terms" in verdict.analysis
    assert (verdict.prompt_tokens, verdict.completion_tokens) == (500, 120)
    assert client.summary()["requests"] == 1
    assert "cell 0 (code)" in server.requests[0]["messages"][0]["content"]


def test_assess_combination_lists_parts(notebook):
    combo = dict(VERDICT, backend="combination", parts=[
        {"part": "assembly", "backend": "cuda"}, {"part": "solve", "backend": "mpi"},
    ])
    with served(reply(combo)) as server:
        verdict, _ = assess_with(server, notebook)
    assert verdict.backend == "combination"
    assert [(p.part, p.backend) for p in verdict.parts] == [("assembly", "cuda"), ("solve", "mpi")]
    assert verdict.chosen_backend == "cuda"
    assert "assembly: cuda" in verdict.summary_line()


def test_assess_accepts_the_july_field_name(notebook):
    old = {"primary": "OpenMP", "also_suitable": ["mpi"], "poor_fit": "cuda"}
    with served("Analysis. " + json.dumps(old)) as server:
        verdict, _ = assess_with(server, notebook)
    assert verdict.backend == "openmp" and verdict.poor_fit == ["cuda"]


@pytest.mark.parametrize("text", [
    "I think OpenMP would be best, but I will not give JSON.",
    '{"backend": "fortran"}',
    '{"backend": "combination", "parts": []}',
    '{"backend": "openmp", ',
])
def test_assess_unparsable_reply_is_a_plain_error(notebook, text):
    with served(text) as server, pytest.raises(ParalleliseError) as caught:
        assess_with(server, notebook)
    assert "usable JSON verdict" in str(caught.value)


def test_assess_rejects_a_bad_notebook(tmp_path):
    bad = tmp_path / "bad.ipynb"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(ParalleliseError, match="not a Jupyter notebook"):
        assess.notebook_text(bad)
    with pytest.raises(ParalleliseError, match="cannot be read"):
        assess.notebook_text(tmp_path / "missing.ipynb")


def test_assess_command_prints_and_writes_json(notebook, tmp_path, capsys):
    out = tmp_path / "sub" / "verdict.json"
    with served(reply(VERDICT)) as server:
        code = cli.main(["assess", str(notebook), "--gateway-url", server.url,
                         "--model", "m1", "--out", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "reduction over independent terms" in printed
    assert "The recommended backend is openmp" in printed
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["backend"] == "openmp" and saved["model"] == "m1"
    assert saved["analysis"].startswith("The loop")


def test_assess_command_without_out_writes_nothing(notebook, tmp_path):
    with served(reply(VERDICT)) as server:
        assert cli.main(["assess", str(notebook), "--gateway-url", server.url]) == 0
    assert not list(tmp_path.glob("*.json"))


def test_assess_command_unparsable_exits_1(notebook, capsys):
    with served("no verdict here") as server:
        code = cli.main(["assess", str(notebook), "--gateway-url", server.url])
    assert code == 1
    err = capsys.readouterr().err
    assert "parallelise: The model" in err and "Traceback" not in err


# --- run --backend auto --------------------------------------------------------------


def auto_setup(monkeypatch, tmp_path, backends=("openmp", "mpi")):
    calls = install_fakes(monkeypatch, tmp_path)
    problem = tmp_path / "problems" / "integrate"
    make_notebook(problem / "integrate.ipynb", "x = 1\n")
    for backend in backends:
        (problem / backend).mkdir()
    fakes_lazy = cli._lazy

    def lazy(name, hint=""):
        return assess if name == "assess" else fakes_lazy(name, hint)

    monkeypatch.setattr(cli, "_lazy", lazy)
    return calls


def auto_args(tmp_path, server):
    return [
        "run", "integrate", "auto", "--agent", "basic", "--model", "m1",
        "--gateway-url", server.url,
        "--problems-dir", str(tmp_path / "problems"),
        "--work-root", str(tmp_path / "work"),
        "--results", str(tmp_path / "runs.jsonl"),
    ]


def test_auto_runs_the_recommended_backend(monkeypatch, tmp_path, capsys):
    calls = auto_setup(monkeypatch, tmp_path)
    with served(reply(dict(VERDICT, backend="mpi"))) as server:
        assert cli.main(auto_args(tmp_path, server)) == 0
    assert calls["assembled"][1] == "mpi"
    work_dir = calls["assembled"][2]
    saved = json.loads((work_dir / "assessment.json").read_text(encoding="utf-8"))
    assert saved["backend"] == "mpi"
    (record,) = ResultsStore(tmp_path / "runs.jsonl").load()
    assert record.backend == "mpi"
    assert "chosen by the assessment" in record.notes and "assessment.json" in record.notes
    assert (record.requests, record.prompt_tokens) == (3, 300)  # the agent's own counts only
    assert "recommended backend is mpi" in capsys.readouterr().out


def test_auto_serial_stops_without_a_record(monkeypatch, tmp_path, capsys):
    calls = auto_setup(monkeypatch, tmp_path)
    serial = dict(VERDICT, backend="serial", justification="The recurrence is sequential.")
    with served(reply(serial)) as server:
        assert cli.main(auto_args(tmp_path, server)) == 1
    err = capsys.readouterr().err
    assert "serial" in err and "no record was written" in err
    assert "assembled" not in calls
    assert not (tmp_path / "runs.jsonl").exists()


def test_auto_backend_the_problem_lacks_stops_without_a_record(monkeypatch, tmp_path, capsys):
    calls = auto_setup(monkeypatch, tmp_path)
    with served(reply(dict(VERDICT, backend="cuda"))) as server:
        assert cli.main(auto_args(tmp_path, server)) == 1
    err = capsys.readouterr().err
    assert "recommends cuda" in err and "no cuda folder" in err and "openmp, mpi" in err
    assert "assembled" not in calls
    assert not (tmp_path / "runs.jsonl").exists()


def test_auto_with_unparsable_verdict_stops_without_a_record(monkeypatch, tmp_path):
    calls = auto_setup(monkeypatch, tmp_path)
    with served("I cannot say.") as server:
        assert cli.main(auto_args(tmp_path, server)) == 1
    assert "assembled" not in calls and not (tmp_path / "runs.jsonl").exists()


def test_a_named_backend_makes_no_assessment_call(monkeypatch, tmp_path):
    auto_setup(monkeypatch, tmp_path)
    with served(reply(VERDICT)) as server:
        args = auto_args(tmp_path, server)
        args[2] = "openmp"
        assert cli.main(args) == 0
        assert server.requests == []


