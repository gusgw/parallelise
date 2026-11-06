"""Tests for the facts noted about the code and the machine."""

from parallelise.provenance import cores, source_hash


def test_the_core_count_is_a_positive_whole_number_no_larger_than_the_processors():
    import os

    assert isinstance(cores(), int) and 1 <= cores() <= len(os.sched_getaffinity(0))


def test_the_source_hash_is_short_and_changes_with_any_python_file(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    before = source_hash(tmp_path)
    assert len(before) == 12 and source_hash(tmp_path) == before
    (tmp_path / "b.py").write_text("y = 2\n")
    source_hash.cache_clear()
    assert source_hash(tmp_path) != before
