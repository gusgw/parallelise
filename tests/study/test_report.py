"""Tests for the tables and the chart of the study report."""


import pytest

from parallelise.cli import main
from parallelise.errors import ParalleliseError
from parallelise.results import ResultsStore, RunRecord
from parallelise.study.report import build_chart, build_tables, load_prices, wilson


def run(model="m1", backend="openmp", passed=True, speedup=2.0, faster=True, **extra):
    fields = dict(problem="integrate", backend=backend, model=model, agent="basic",
                  correct=passed, failure_stage="none" if passed else "test",
                  speedup=speedup if passed else None, faster=passed and faster,
                  requests=10, prompt_tokens=1000, completion_tokens=100, latency_mean_s=2.0,
                  tokens_per_second=50.0, agent_time_s=60.0, scaling=1.8)  # fmt: skip
    return RunRecord(**{**fields, **extra})


def test_every_run_counts_repeats_included():
    records = [run(), run(), run(passed=False)]
    assert "2/3" in build_tables(records)


def test_the_wilson_interval_stays_inside_zero_and_one():
    assert wilson(0, 5)[0] == 0.0 and wilson(5, 5)[1] == 1.0
    low, high = wilson(5, 10)
    assert 0.2 < low < 0.5 < high < 0.8


def test_speed_is_reported_for_correct_runs_only():
    tables = build_tables([run(speedup=3.0), run(speedup=1.0, faster=False),
                           run(passed=False)])  # fmt: skip
    assert "| integrate | openmp | m1 | 2 | 1 | 2.00 | 1.80 |" in tables


def test_the_model_table_gives_latency_rate_and_cost(tmp_path):
    prices = tmp_path / "prices.json"
    prices.write_text('{"source": "a note", "usd_per_million_tokens": '
                      '{"m1": {"input": 1.0, "output": 10.0}}}')  # fmt: skip
    tables = build_tables([run(), run()], load_prices(prices))
    assert "| m1 | 2 | 20 | 2.00 | 50 | 2000 | 200 | 0.0 | 0.00 |" in tables


BAD_PRICES = ["not json", '{"m1": {"input": 1}}', '{"m1": {"input": "x", "output": 1}}']


@pytest.mark.parametrize("text", BAD_PRICES)
def test_a_price_file_that_cannot_be_used_is_refused(tmp_path, text):
    path = tmp_path / "prices.json"
    path.write_text(text)
    with pytest.raises(ParalleliseError):
        load_prices(path)


def test_the_chart_is_written_as_svg(tmp_path):
    pytest.importorskip("matplotlib")
    dest = build_chart([run(), run(backend="mpi", passed=False)], tmp_path / "chart.svg")
    assert dest.read_text().lstrip().startswith("<?xml")


def test_the_report_command_reads_a_results_file(tmp_path, capsys):
    store = ResultsStore(tmp_path / "runs.jsonl")
    store.append(run())
    assert main(["report", "--results", str(tmp_path / "runs.jsonl")]) == 0
    assert "1/1" in capsys.readouterr().out


def test_the_report_command_says_when_there_are_no_runs(tmp_path, capsys):
    assert main(["report", "--results", str(tmp_path / "none.jsonl")]) == 1
    assert "no run records" in capsys.readouterr().err


def test_no_records_give_a_plain_sentence():
    assert build_tables([]) == "There are no runs to report."


def test_a_run_whose_notebook_failed_is_left_out_and_said_to_be():
    tables = build_tables([run(), run(passed=False, failure_stage="notebook")])
    assert "1/1" in tables and "1 run is left out" in tables
