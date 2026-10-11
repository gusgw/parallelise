"""Tests for the benchmark table checker, using fabricated and genuine tables."""

from pathlib import Path

import pytest

from parallelise.bench_check import check_bench_csv, check_bench_cuda, main

HEADER = "threads,time_s,speedup,efficiency,throughput\n"


def write(tmp_path: Path, rows: list[tuple], header: str = HEADER) -> Path:
    path = tmp_path / "bench.csv"
    path.write_text(header + "".join(",".join(str(v) for v in r) + "\n" for r in rows))
    return path


GENUINE = [
    (1, 1.000000, 1.000, 1.000, 50.0),
    (2, 0.540000, 1.852, 0.926, 92.6),
    (4, 0.310000, 3.226, 0.806, 161.3),
    (8, 0.190000, 5.263, 0.658, 263.2),
]


def test_genuine_table_is_accepted(tmp_path):
    verdict = check_bench_csv(write(tmp_path, GENUINE), floor=1.5)
    assert verdict.ok, verdict.reasons
    assert verdict.speedup_best == pytest.approx(5.263)


def test_speedup_disagreeing_with_times_is_rejected(tmp_path):
    rows = [GENUINE[0], (2, 0.54, 1.95, 0.975, 92.6), GENUINE[2], GENUINE[3]]
    verdict = check_bench_csv(write(tmp_path, rows), floor=1.5)
    assert not verdict.ok
    assert any("disagrees with the times" in r for r in verdict.reasons)


def test_times_exactly_t1_over_width_are_rejected(tmp_path):
    rows = [
        (1, 1.0, 1.0, 1.0, 50.0),
        (2, 0.5, 2.0, 1.0, 100.0),
        (4, 0.25, 4.0, 1.0, 200.0),
        (8, 0.125, 8.0, 1.0, 400.0),
    ]
    verdict = check_bench_csv(write(tmp_path, rows), floor=1.5)
    assert not verdict.ok
    assert any("exactly t1" in r for r in verdict.reasons)


def test_duplicate_widths_are_rejected(tmp_path):
    rows = [GENUINE[0], GENUINE[1], GENUINE[1], GENUINE[3]]
    verdict = check_bench_csv(write(tmp_path, rows), floor=1.5)
    assert not verdict.ok
    assert any("duplicate" in r for r in verdict.reasons)


def test_constant_throughput_is_rejected(tmp_path):
    rows = [(w, t, s, e, 77.0) for (w, t, s, e, _) in GENUINE]
    verdict = check_bench_csv(write(tmp_path, rows), floor=1.5)
    assert not verdict.ok
    assert any("throughput" in r for r in verdict.reasons)


def test_efficiency_not_one_at_one_worker_is_rejected(tmp_path):
    rows = [(1, 1.0, 1.0, 0.5, 50.0)] + GENUINE[1:]
    verdict = check_bench_csv(write(tmp_path, rows), floor=1.5)
    assert not verdict.ok
    assert any("efficiency at 1" in r for r in verdict.reasons)


def test_speedup_not_one_at_one_worker_is_rejected(tmp_path):
    rows = [(1, 2.0, 0.5, 1.0, 50.0), (2, 1.0, 2.0, 1.0, 90.0)]
    verdict = check_bench_csv(write(tmp_path, rows), floor=1.5)
    assert not verdict.ok
    assert any("speedup at 1" in r for r in verdict.reasons)


def test_best_speedup_below_floor_is_rejected(tmp_path):
    rows = [
        (1, 1.00, 1.000, 1.000, 50.0),
        (2, 0.98, 1.020, 0.510, 51.0),
        (4, 0.95, 1.053, 0.263, 52.6),
    ]
    verdict = check_bench_csv(write(tmp_path, rows), floor=1.5)
    assert not verdict.ok
    assert any("below" in r for r in verdict.reasons)
    assert check_bench_csv(write(tmp_path, rows), floor=1.0).ok


def test_implausible_speedup_is_rejected(tmp_path):
    rows = [(1, 1.0, 1.0, 1.0, 50.0), (2, 0.2, 5.0, 2.5, 250.0)]
    verdict = check_bench_csv(write(tmp_path, rows), floor=1.5)
    assert not verdict.ok
    assert any("implausible" in r for r in verdict.reasons)


def test_missing_column_and_missing_file_are_rejected(tmp_path):
    bad = write(tmp_path, [(1, 1.0), (2, 0.6)], header="threads,time_s\n")
    assert not check_bench_csv(bad, floor=1.5).ok
    assert not check_bench_csv(tmp_path / "absent.csv", floor=1.5).ok


def test_mpi_ranks_header_is_understood(tmp_path):
    path = write(tmp_path, GENUINE, header=HEADER.replace("threads", "ranks"))
    assert check_bench_csv(path, floor=1.5).ok


def test_floor_comes_from_environment(tmp_path, monkeypatch):
    path = write(tmp_path, GENUINE)
    monkeypatch.setenv("SPEEDUP_FLOOR", "9")
    assert not check_bench_csv(path).ok
    monkeypatch.setenv("SPEEDUP_FLOOR", "2")
    assert check_bench_csv(path).ok


def test_command_line_exit_codes(tmp_path, capsys):
    assert main([str(write(tmp_path, GENUINE))]) == 0
    assert "bench stats OK" in capsys.readouterr().out
    bad = write(tmp_path, [(1, 1.0, 1.0, 1.0, 5.0), (2, 0.99, 1.01, 0.5, 5.1)])
    assert main([str(bad)]) == 1
    assert "BENCH CHECK FAIL" in capsys.readouterr().out
    assert main([]) == 2


def cuda_table(tmp_path: Path, serial: tuple, gpu: tuple) -> Path:
    path = tmp_path / "cuda.csv"
    path.write_text(f"impl,time_s,speedup\nserial,{serial[0]},{serial[1]}\ngpu,{gpu[0]},{gpu[1]}\n")
    return path


def test_cuda_genuine_and_fabricated(tmp_path):
    assert check_bench_cuda(cuda_table(tmp_path, (2.0, 1.0), (0.1, 20.0)), floor=5).ok
    assert not check_bench_cuda(cuda_table(tmp_path, (2.0, 1.0), (0.1, 40.0)), floor=5).ok
    assert not check_bench_cuda(cuda_table(tmp_path, (2.0, 1.0), (1.0, 2.0)), floor=5).ok
    assert not check_bench_cuda(cuda_table(tmp_path, (2.0, 1.5), (0.1, 20.0)), floor=5).ok
    assert not check_bench_cuda(tmp_path / "absent.csv").ok
