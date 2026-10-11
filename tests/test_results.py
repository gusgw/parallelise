"""Tests for run records and the results store."""

import json

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
    assert not store.path.exists()


def test_damaged_line_names_its_number(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps(make().to_dict()) + "\n{broken\n")
    with pytest.raises(ResultsError, match="Line 2"):
        ResultsStore(path).load()


def test_unknown_keys_are_ignored(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text(json.dumps({**make().to_dict(), "future": 1}) + "\n\n")
    assert len(ResultsStore(path).load()) == 1
