"""Tests for the run record and the file of records."""

import pytest

from parallelise.errors import ParalleliseError
from parallelise.results import ResultsStore, RunRecord


def test_records_are_appended_and_read_back(tmp_path):
    store = ResultsStore(tmp_path / "deep" / "runs.jsonl")
    assert store.load() == []
    first = RunRecord("integrate", "openmp", "m", "basic", correct=True, speedup=2.5, faster=True)
    store.append(first)
    store.append(RunRecord("integrate", "mpi", "m", "basic", correct=False,
                           failure_stage="test"))  # fmt: skip
    assert store.load()[0] == first and store.load()[1].failure_stage == "test"


def test_a_damaged_line_names_its_number(tmp_path):
    path = tmp_path / "runs.jsonl"
    store = ResultsStore(path)
    store.append(RunRecord("p", "openmp", "m", "basic", correct=True))
    with open(path, "a") as handle:
        handle.write("{not json\n")
    with pytest.raises(ParalleliseError, match="Line 2"):
        store.load()


def test_a_file_that_cannot_be_written_is_found_before_the_run(tmp_path):
    with pytest.raises(ParalleliseError, match="cannot be recorded"):
        ResultsStore(tmp_path).ensure_writable()
