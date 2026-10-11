"""Tests for the tables and chart in parallelise.report."""
from __future__ import annotations

import json
from pathlib import Path

from parallelise.cli import main
from parallelise.report import (
    build_chart,
    build_comparison_chart,
    build_comparison_tables,
    build_serving_table,
    build_tables,
    latest_cells,
    load_prices,
)


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


def served(model="m1", requests=10, latency=2.0, speed=50.0, prompt=1000, completion=500,
           wall=60.0, **kwargs) -> dict:
    """Build a record that also carries the serving fields."""
    record = make(model=model, **kwargs)
    record.update(requests=requests, latency_mean_s=latency, tokens_per_second=speed,
                  prompt_tokens=prompt, completion_tokens=completion, wall_time_s=wall)
    return record


def test_comparison_needs_both_kinds_of_run() -> None:
    assert build_comparison_tables([make()]) == ""
    assert build_comparison_tables([make(masked=True)]) == ""


def test_comparison_tables_show_the_change() -> None:
    records = [make(), make(backend="mpi"), make(masked=True),
               make(backend="mpi", masked=True, result="FAIL")]
    text = build_comparison_tables(records)
    assert "| m1 | 2/2 (100%) | 1/2 (50%) | -50 |" in text
    assert "| integrate on mpi | 1/1 (100%) | 0/1 (0%) | -100 |" in text
    assert "| integrate on openmp | 1/1 (100%) | 1/1 (100%) | 0 |" in text
    assert "Masked against unmasked" in build_tables(records)


def test_serving_table_uses_medians_totals_and_prices() -> None:
    records = [served(latency=1.0, speed=40, prompt=2_000_000, completion=1_000_000),
               served(agent="pydantic", latency=3.0, speed=60, prompt=0, completion=0),
               served(agent="langgraph", requests=0, latency=0.0, speed=0.0, prompt=0,
                      completion=0)]
    text = build_serving_table(records, {"m1": (3.0, 15.0)})
    assert "| m1 | 3 | 20 | 2.00 | 50 | 2,000,000 | 1,000,000 | 0.1 | 21.00 |" in text
    assert "| All models | 3 | 20 |" in text
    assert "m1 3 in and 15 out" in text


def test_serving_table_without_prices_has_no_cost_and_omits_old_records() -> None:
    assert "Cost" not in build_serving_table([served()])
    assert build_serving_table([make()]) == ""
    assert "Serving" not in build_tables([make()])


def test_comparison_chart_is_written_as_svg(tmp_path: Path) -> None:
    dest = tmp_path / "masked.svg"
    build_comparison_chart([make(), make(masked=True, result="FAIL")], dest)
    text = dest.read_text()
    assert "Unmasked" in text and "Masked" in text


def test_prices_file_is_read_and_notes_ignored(tmp_path: Path) -> None:
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"source": "a note", "usd_per_million_tokens": {
        "m1": {"input": 1, "output": 2}}}))
    assert load_prices(path) == {"m1": (1.0, 2.0)}


def test_report_command_takes_store_and_prices(tmp_path: Path, capsys) -> None:
    store = tmp_path / "runs.jsonl"
    store.write_text("\n".join(json.dumps(served(masked=m)) for m in (False, True)) + "\n")
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"m1": {"input": 1, "output": 2}}))
    chart = tmp_path / "c.svg"
    assert main(["report", "--store", str(store), "--prices", str(prices),
                 "--masked-chart", str(chart)]) == 0
    out = capsys.readouterr().out
    assert "Serving" in out and "Cost (US dollars)" in out
    assert chart.exists()
