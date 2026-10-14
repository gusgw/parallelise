"""Tests for the study commands as reached through the main command line."""

import sys

from parallelise import cli


def test_unavailable_masking_module_gives_clear_message(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "parallelise.study.mask", None)
    (tmp_path / "problems" / "p").mkdir(parents=True)
    code = cli.main(["mask", str(tmp_path / "problems" / "p"), "abc", str(tmp_path / "d")])
    assert code == 1
    assert "'study.mask' part of parallelise is not available" in capsys.readouterr().err


def test_report_with_no_records(tmp_path, capsys):
    assert cli.main(["report", "--results", str(tmp_path / "none.jsonl")]) == 1
    assert "no run records" in capsys.readouterr().err
