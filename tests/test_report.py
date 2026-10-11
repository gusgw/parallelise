"""Tests for the tables and chart in parallelise.report."""
from __future__ import annotations

import json
from pathlib import Path

from parallelise.report import build_chart, build_tables, latest_cells


def make(problem="integrate", backend="openmp", model="m1", agent="basic",
         result="PASS", when="2026-07-04T10:00:00+10:00", masked=False) -> dict:
    """Build a minimal record with the fields the report reads."""
    return {"problem": problem, "backend": backend, "model": model, "agent": agent,
            "result": result, "when": when, "masked": masked}


def test_latest_run_of_a_cell_replaces_earlier_ones() -> None:
    records = [make(result="FAIL", when="2026-07-04T10:00:00+10:00"),
               make(result="PASS", when="2026-07-05T10:00:00+10:00")]
    cells = latest_cells(records)
    assert len(cells) == 1 and cells[0]["result"] == "PASS"


def test_latest_is_judged_by_time_not_by_order() -> None:
    records = [make(result="PASS", when="2026-07-05T10:00:00+10:00"),
               make(result="FAIL", when="2026-07-04T10:00:00+10:00")]
    assert latest_cells(records)[0]["result"] == "PASS"


def test_masked_runs_are_kept_apart() -> None:
    records = [make(), make(masked=True, result="FAIL")]
    assert len(latest_cells(records)) == 1
    assert latest_cells(records, masked=True)[0]["result"] == "FAIL"


def test_tables_count_passes_and_state_the_rule() -> None:
    records = [make(), make(backend="mpi", result="FAIL"),
               make(agent="pydantic"), make(model="m2", result="FAIL")]
    text = build_tables(records)
    assert "only the latest run of each cell is counted" in text
    assert "| m1 | 2/2 (100%) | 0/1 (0%) | 2/3 (67%) |" in text
    assert "| m2 | 0/1 (0%) | - | 0/1 (0%) |" in text
    assert "### Passes by problem and model" in text


def test_tables_with_no_records() -> None:
    assert "No runs" in build_tables([])


def test_chart_is_written_as_svg(tmp_path: Path) -> None:
    dest = tmp_path / "figures" / "chart.svg"
    build_chart([make(), make(backend="mpi", result="FAIL")], dest)
    assert dest.read_text().lstrip().startswith(("<?xml", "<svg"))
    assert "Pass rate" in dest.read_text()


def test_july_records_give_228_cells() -> None:
    path = Path(__file__).resolve().parent.parent / "results" / "runs.jsonl"
    if not path.exists():
        return
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(latest_cells(records)) == 228
