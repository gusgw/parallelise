"""Tests for the tables and chart in parallelise.study.report."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from parallelise.cli import main
from parallelise.errors import ResultsError
from parallelise.results import ResultsStore, RunRecord
from parallelise.study.report import (
    build_chart,
    build_comparison_chart,
    build_comparison_tables,
    build_serving_table,
    build_speed_table,
    build_tables,
    latest_cells,
    load_prices,
    load_reference,
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


def test_tables_ignore_the_fields_added_to_records_later() -> None:
    newer = RunRecord(
        problem="integrate", backend="openmp", model="m1", agent="basic", result="PASS",
        requests=10, prompt_tokens=1000, completion_tokens=500, wall_time_s=60.0,
        latency_mean_s=2.0, tokens_per_second=50.0, temperature=0.2, step_budget=80,
        time_limit_s=1800.0, agent_time_s=50.0, check_time_s=10.0,
        framework_versions={"httpx": "0.28.1"}, cpu_count=12, package_version="0.1.0",
        source_hash="0123456789ab",
    )
    older = make(problem="matmul", model="m1")  # a dictionary with only the original fields
    for records in ([newer], [older], [newer.to_dict(), older], [newer, older]):
        text = build_tables(records)
        assert "### Passes by model and backend" in text
    assert "Serving" in build_tables([newer, older])


def test_report_command_reads_a_store_of_new_records(tmp_path: Path, capsys) -> None:
    store = ResultsStore(tmp_path / "runs.jsonl")
    store.append(RunRecord(problem="a", backend="openmp", model="m", agent="basic",
                           result="PASS", framework_versions={"openai": "3.29.0"}, cpu_count=4))
    assert main(["study", "report", "--results", str(store.path)]) == 0
    assert "1/1 (100%)" in capsys.readouterr().out


def test_a_bad_timestamp_or_number_is_a_results_error() -> None:
    with pytest.raises(ResultsError, match="yesterday"):
        latest_cells([make(when="yesterday")])
    with pytest.raises(ResultsError, match="requests"):
        build_serving_table([served(requests="lots")])


@pytest.mark.parametrize("text", [None, "{oops", "[1, 2]", '{"m": {"input": 1}}',
                                  '{"m": {"input": "x", "output": 2}}'])
def test_a_malformed_price_file_is_a_results_error(tmp_path: Path, text) -> None:
    path = tmp_path / "prices.json"
    if text is not None:
        path.write_text(text)
    with pytest.raises(ResultsError, match="prices.json"):
        load_prices(path)


# --- the speed of correct routines ------------------------------------------------------------


def timed(speed: str | None, best: float | None, **kwargs) -> dict:
    """Build a record with a speed verdict and a best speed-up; None leaves the verdict out."""
    record = make(**kwargs)
    record["speedup_best"] = best
    if speed is not None:
        record["speed_verdict"] = speed
    return record


SPEED_RECORDS = [
    timed("faster", 3.0, agent="basic"),
    timed("no useful speed-up", 1.1, agent="pydantic"),
    timed("faster", 2.0, agent="langgraph"),
    timed("speed not judged on this machine", 9.0, agent="other"),  # left out of the median
    timed(None, 1.5, agent="older"),  # a record made before the verdict was kept
    timed("not measured", None, agent="broken", result="FAIL"),  # not correct: left out
    timed("no useful speed-up", 1.07, problem="scan", backend="mpi"),
]


def test_speed_table_counts_each_verdict_and_gives_the_median() -> None:
    text = build_speed_table(SPEED_RECORDS)
    assert "### Speed of correct routines, by problem, backend and model" in text
    assert ("| Problem | Backend | Model | Correct runs | Faster | No useful speed-up | "
            "Not judged | Not measured | Not recorded | Median best speed-up |") in text
    assert "| integrate | openmp | m1 | 5 | 2 | 1 | 1 | 0 | 1 | 1.75 |" in text
    assert "| scan | mpi | m1 | 1 | 0 | 1 | 0 | 0 | 0 | 1.07 |" in text
    assert "| --- | --- | --- | ---: |" in text  # the three names are aligned left
    assert "Known-good" not in text


def test_speed_table_shows_the_known_good_speedup_beside_each_model() -> None:
    reference = {("scan", "mpi"): 1.1, ("integrate", "openmp"): 7.5}
    records = [*SPEED_RECORDS, timed("faster", 6.0, model="m2")]
    text = build_speed_table(records, reference=reference)
    assert text.splitlines()[4].endswith("| Median best speed-up | Known-good speed-up |")
    assert "| integrate | openmp | m1 | 5 | 2 | 1 | 1 | 0 | 1 | 1.75 | 7.50 |" in text
    assert "| integrate | openmp | m2 | 1 | 1 | 0 | 0 | 0 | 0 | 6.00 | 7.50 |" in text
    assert "| scan | mpi | m1 | 1 | 0 | 1 | 0 | 0 | 0 | 1.07 | 1.10 |" in text
    unknown = build_speed_table([timed("faster", 2.0, problem="cg")], reference=reference)
    assert unknown.splitlines()[-1].endswith("| 2.00 | - |")


def test_speed_table_is_left_out_when_no_run_is_correct() -> None:
    assert build_speed_table([make(result="FAIL")]) == ""
    assert "Speed of correct routines" not in build_tables([make(result="FAIL")])


def test_records_without_a_speed_verdict_do_not_break_the_report() -> None:
    older = make()  # a dictionary with only the original fields: no speedup_best either
    newer = RunRecord(problem="integrate", backend="openmp", model="m1", agent="pydantic",
                      result="PASS", speedup_best=1.07)  # speed_verdict is None
    text = build_tables([older, newer, newer.to_dict() | {"agent": "x"}])
    assert "| integrate | openmp | m1 | 3 | 0 | 0 | 0 | 0 | 3 | 1.07 |" in text


def test_correctness_tables_count_a_correct_routine_that_is_not_faster_as_a_pass() -> None:
    text = build_tables([timed("no useful speed-up", 1.07)])
    assert "| m1 | 1/1 (100%) | 1/1 (100%) |" in text
    assert "A pass means a correct routine, whatever its speed" in text


def write_reference(tmp_path: Path, data) -> Path:
    path = tmp_path / "reference.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


def test_reference_file_is_read_and_other_keys_ignored(tmp_path: Path) -> None:
    path = write_reference(tmp_path, [
        {"problem": "scan", "backend": "mpi", "best_speedup": 1.1, "note": "four processes"},
        {"problem": "integrate", "backend": "openmp", "best_speedup": 7},
    ])
    assert load_reference(path) == {("scan", "mpi"): 1.1, ("integrate", "openmp"): 7.0}


@pytest.mark.parametrize("data", [
    None, "{oops", {"scan": 1.1}, [1, 2], [{"problem": "scan", "backend": "mpi"}],
    [{"problem": "scan", "backend": "mpi", "best_speedup": "fast"}],
    [{"problem": "scan", "backend": "mpi", "best_speedup": True}],
], ids=["missing", "not-json", "object", "numbers", "no-speedup", "text", "true"])
def test_a_malformed_reference_file_is_a_results_error(tmp_path: Path, data) -> None:
    path = tmp_path / "reference.json" if data is None else write_reference(tmp_path, data)
    with pytest.raises(ResultsError, match="reference.json"):
        load_reference(path)


def test_report_command_takes_a_reference_file(tmp_path: Path, capsys) -> None:
    store = tmp_path / "runs.jsonl"
    store.write_text(json.dumps(timed("no useful speed-up", 1.07, problem="scan",
                                      backend="mpi")) + "\n")
    reference = write_reference(tmp_path, [{"problem": "scan", "backend": "mpi",
                                            "best_speedup": 1.1}])
    assert main(["study", "report", "--results", str(store), "--reference",
                 str(reference)]) == 0
    out = capsys.readouterr().out
    assert "| scan | mpi | m1 | 1 | 0 | 1 | 0 | 0 | 0 | 1.07 | 1.10 |" in out
    assert main(["study", "report", "--results", str(store)]) == 0
    assert "Known-good" not in capsys.readouterr().out
    assert main(["study", "report", "--results", str(store), "--reference",
                 str(tmp_path / "absent.json")]) == 1
    assert "absent.json" in capsys.readouterr().err
