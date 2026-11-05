"""Tests for run records and the results store."""

import json
from pathlib import Path

import pytest

from parallelise.errors import ResultsError
from parallelise.results import ResultsStore, RunRecord


def make(**kw):
    base = dict(problem="integrate", backend="openmp", model="m", agent="basic", result="PASS")
    base.update(kw)
    return RunRecord(**base)


def test_round_trip(tmp_path):
    store = ResultsStore(tmp_path / "deep" / "runs.jsonl")
    assert store.load() == []
    a = make(speedup_best=3.5, notes="fine")
    b = make(result="FAIL", failure_stage="build")
    store.append(a)
    store.append(b)
    assert store.load() == [a, b]
    assert len(store.path.read_text().splitlines()) == 2


def test_all_contract_fields_are_present():
    keys = set(make().to_dict())
    assert keys == {
        "when", "problem", "backend", "model", "agent", "agent_version", "masked", "result",
        "failure_stage", "speedup_best", "wall_time_s", "requests", "prompt_tokens",
        "completion_tokens", "latency_mean_s", "tokens_per_second", "serving", "quantisation",
        "notes",
        # added later; every one has a default so that older records still load
        "temperature", "step_budget", "time_limit_s", "agent_time_s", "check_time_s",
        "framework_versions", "cpu_count", "package_version", "source_hash", "speed_verdict",
    }
    assert make().speedup_best is None
    assert make().serving == "not recorded"
    assert make().quantisation == "not disclosed"


def test_invalid_values_are_refused(tmp_path):
    store = ResultsStore(tmp_path / "r.jsonl")
    with pytest.raises(ResultsError):
        store.append(make(result="MAYBE"))
    with pytest.raises(ResultsError):
        store.append(make(failure_stage="sulking"))
    with pytest.raises(ResultsError, match="speed verdict"):
        store.append(make(speed_verdict="quite quick"))
    assert not store.path.exists()


@pytest.mark.parametrize(
    "verdict",
    [None, "faster", "no useful speed-up", "speed not judged on this machine", "not measured"],
)
def test_every_speed_verdict_round_trips(tmp_path, verdict):
    store = ResultsStore(tmp_path / "r.jsonl")
    record = make(speed_verdict=verdict, speedup_best=1.07)
    store.append(record)
    assert store.load() == [record]


def test_damaged_line_names_its_number(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(make().to_dict()) + "\n{broken\n")
    with pytest.raises(ResultsError, match="Line 2"):
        ResultsStore(path).load()


def test_unknown_keys_are_ignored(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({**make().to_dict(), "future": 1}) + "\n\n")
    assert len(ResultsStore(path).load()) == 1


# --- the fuller records ------------------------------------------------------------------------

NEW_FIELDS = {
    "temperature": None, "step_budget": None, "time_limit_s": None, "agent_time_s": None,
    "check_time_s": None, "framework_versions": {}, "cpu_count": None, "package_version": "",
    "source_hash": "", "speed_verdict": None,
}


def test_the_new_fields_default_to_not_recorded():
    record = make()
    for name, default in NEW_FIELDS.items():
        assert getattr(record, name) == default, name


def test_new_fields_round_trip(tmp_path):
    store = ResultsStore(tmp_path / "runs.jsonl")
    record = make(
        temperature=0.2, step_budget=80, time_limit_s=1800.0, agent_time_s=61.5,
        check_time_s=12.25, framework_versions={"httpx": "0.28.1", "openai": "3.29.0"},
        cpu_count=12, package_version="0.1.0", source_hash="0123456789ab",
    )
    store.append(record)
    assert store.load() == [record]
    assert json.loads(store.path.read_text())["framework_versions"] == {
        "httpx": "0.28.1", "openai": "3.29.0"}


def test_a_record_written_before_the_new_fields_still_loads(tmp_path):
    old = {
        "problem": "scan", "backend": "openmp", "model": "kimi-k2.5", "agent": "pydantic",
        "result": "FAIL", "failure_stage": "test-one-worker", "when": "2026-10-11T20:00:16+00:00",
        "agent_version": "d008f87f9e", "masked": True, "speedup_best": None,
        "wall_time_s": 42.13, "requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "latency_mean_s": 0.0, "tokens_per_second": 0.0, "serving": "Amazon Bedrock",
        "quantisation": "not disclosed by the provider", "notes": "an older note",
    }
    path = tmp_path / "old.jsonl"
    path.write_text(json.dumps(old) + "\n")
    (record,) = ResultsStore(path).load()
    assert record.failure_stage == "test-one-worker" and record.wall_time_s == 42.13
    for name, default in NEW_FIELDS.items():
        assert getattr(record, name) == default, name


def test_a_null_framework_versions_is_read_as_empty(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({**make().to_dict(), "framework_versions": None}) + "\n")
    assert ResultsStore(path).load()[0].framework_versions == {}


REPOSITORY = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("name", ["results/runs.jsonl", "results/campaign-2026-10/runs.jsonl"])
def test_the_records_already_kept_in_the_repository_still_load(name):
    path = REPOSITORY / name
    if not path.exists():
        pytest.skip(f"{name} is not in this checkout")
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    records = ResultsStore(path).load()
    assert len(records) == len(lines) > 0
    assert all(r.package_version == "" and r.cpu_count is None for r in records)
    assert all(r.speed_verdict is None for r in records)


def test_the_july_import_script_still_writes_records_the_store_reads():
    import importlib.util

    script = REPOSITORY / "scripts" / "import_july_runs.py"
    manifest = REPOSITORY / "results" / "manifest-july-2026.tsv"
    if not script.exists() or not manifest.exists():
        pytest.skip("the July import script or its manifest is not in this checkout")
    spec = importlib.util.spec_from_file_location("import_july_runs", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = module.build_records()
    assert records and all(tuple(r) == module.FIELDS for r in records)
    loaded = [RunRecord.from_dict(r) for r in records]
    assert all(r.temperature is None and r.source_hash == "" for r in loaded)
