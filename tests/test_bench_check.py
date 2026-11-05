"""Tests for the benchmark table checker, using fabricated and genuine tables."""

from pathlib import Path

import pytest

from parallelise.bench_check import check_cpu_table, check_gpu_table, main

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
    verdict = check_cpu_table(write(tmp_path, GENUINE), floor=1.5)
    assert verdict.ok, verdict.reasons
    assert verdict.speedup_best == pytest.approx(5.263)


def test_speedup_disagreeing_with_times_is_rejected(tmp_path):
    rows = [GENUINE[0], (2, 0.54, 1.95, 0.975, 92.6), GENUINE[2], GENUINE[3]]
    verdict = check_cpu_table(write(tmp_path, rows), floor=1.5)
    assert not verdict.ok
    assert any("disagrees with the times" in r for r in verdict.reasons)


def test_times_exactly_t1_over_width_are_rejected(tmp_path):
    rows = [
        (1, 1.0, 1.0, 1.0, 50.0),
        (2, 0.5, 2.0, 1.0, 100.0),
        (4, 0.25, 4.0, 1.0, 200.0),
        (8, 0.125, 8.0, 1.0, 400.0),
    ]
    verdict = check_cpu_table(write(tmp_path, rows), floor=1.5)
    assert not verdict.ok
    assert any("exactly t1" in r for r in verdict.reasons)


def test_duplicate_widths_are_rejected(tmp_path):
    rows = [GENUINE[0], GENUINE[1], GENUINE[1], GENUINE[3]]
    verdict = check_cpu_table(write(tmp_path, rows), floor=1.5)
    assert not verdict.ok
    assert any("duplicate" in r for r in verdict.reasons)


def test_constant_throughput_is_rejected(tmp_path):
    rows = [(w, t, s, e, 77.0) for (w, t, s, e, _) in GENUINE]
    verdict = check_cpu_table(write(tmp_path, rows), floor=1.5)
    assert not verdict.ok
    assert any("throughput" in r for r in verdict.reasons)


def test_efficiency_not_one_at_one_worker_is_rejected(tmp_path):
    rows = [(1, 1.0, 1.0, 0.5, 50.0)] + GENUINE[1:]
    verdict = check_cpu_table(write(tmp_path, rows), floor=1.5)
    assert not verdict.ok
    assert any("efficiency at 1" in r for r in verdict.reasons)


def test_speedup_not_one_at_one_worker_is_rejected(tmp_path):
    rows = [(1, 2.0, 0.5, 1.0, 50.0), (2, 1.0, 2.0, 1.0, 90.0)]
    verdict = check_cpu_table(write(tmp_path, rows), floor=1.5)
    assert not verdict.ok
    assert any("speedup at 1" in r for r in verdict.reasons)


def test_best_speedup_below_floor_is_rejected(tmp_path):
    rows = [
        (1, 1.00, 1.000, 1.000, 50.0),
        (2, 0.98, 1.020, 0.510, 51.0),
        (4, 0.95, 1.053, 0.263, 52.6),
    ]
    verdict = check_cpu_table(write(tmp_path, rows), floor=1.5)
    assert not verdict.ok
    assert any("below" in r for r in verdict.reasons)
    assert check_cpu_table(write(tmp_path, rows), floor=1.0).ok


def test_implausible_speedup_is_rejected(tmp_path):
    rows = [(1, 1.0, 1.0, 1.0, 50.0), (2, 0.2, 5.0, 2.5, 250.0)]
    verdict = check_cpu_table(write(tmp_path, rows), floor=1.5)
    assert not verdict.ok
    assert any("implausible" in r for r in verdict.reasons)


def test_missing_column_and_missing_file_are_rejected(tmp_path):
    bad = write(tmp_path, [(1, 1.0), (2, 0.6)], header="threads,time_s\n")
    assert not check_cpu_table(bad, floor=1.5).ok
    assert not check_cpu_table(tmp_path / "absent.csv", floor=1.5).ok


def test_mpi_ranks_header_is_understood(tmp_path):
    path = write(tmp_path, GENUINE, header=HEADER.replace("threads", "ranks"))
    assert check_cpu_table(path, floor=1.5).ok


def test_floor_comes_from_environment(tmp_path, monkeypatch):
    path = write(tmp_path, GENUINE)
    monkeypatch.setenv("SPEEDUP_FLOOR", "9")
    assert not check_cpu_table(path).ok
    monkeypatch.setenv("SPEEDUP_FLOOR", "2")
    assert check_cpu_table(path).ok


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
    assert check_gpu_table(cuda_table(tmp_path, (2.0, 1.0), (0.1, 20.0)), floor=5).ok
    assert not check_gpu_table(cuda_table(tmp_path, (2.0, 1.0), (0.1, 40.0)), floor=5).ok
    assert not check_gpu_table(cuda_table(tmp_path, (2.0, 1.0), (1.0, 2.0)), floor=5).ok
    assert not check_gpu_table(cuda_table(tmp_path, (2.0, 1.5), (0.1, 20.0)), floor=5).ok
    assert not check_gpu_table(tmp_path / "absent.csv").ok


def test_old_names_still_work(tmp_path):
    from parallelise import bench_check

    assert bench_check.check_bench_csv(write(tmp_path, GENUINE), floor=1.5).ok
    table = cuda_table(tmp_path, (2.0, 1.0), (0.1, 20.0))
    assert bench_check.check_bench_cuda(table, floor=3).ok


# --- the three kinds of rule, kept apart in the verdict -------------------------------------

SERIAL = [
    (1, 1.00, 1.000, 1.000, 50.0),
    (2, 0.98, 1.020, 0.510, 51.0),
    (4, 0.95, 1.053, 0.263, 52.6),
]
INVENTED = [
    (1, 1.0, 1.0, 1.0, 50.0),
    (2, 0.5, 2.0, 1.0, 100.0),
    (4, 0.25, 4.0, 1.0, 200.0),
]


@pytest.fixture(autouse=True)
def plain_environment(monkeypatch):
    for name in ("SPEEDUP_FLOOR", "BENCH_JUDGE_SPEED", "BENCH_FLOOR_ROLE"):
        monkeypatch.delenv(name, raising=False)


def test_a_faster_table_is_sound_and_reaches_the_floor(tmp_path):
    verdict = check_cpu_table(write(tmp_path, GENUINE), floor=1.5)
    assert (verdict.ok, verdict.sound, verdict.form_ok) == (True, True, True)
    assert verdict.invented == []
    assert (verdict.floor, verdict.floor_reached) == (1.5, True)
    assert verdict.speed_verdict == "faster"


def test_a_serial_table_is_sound_but_gives_no_useful_speedup(tmp_path):
    verdict = check_cpu_table(write(tmp_path, SERIAL), floor=1.5)
    assert verdict.sound and verdict.form_ok and verdict.invented == []
    assert verdict.floor_reached is False
    assert verdict.speed_verdict == "no useful speed-up"
    assert verdict.speedup_best == pytest.approx(1.053)
    assert not verdict.ok  # the agent's own make check still fails on it
    assert any("below" in r for r in verdict.reasons)


def test_a_best_speedup_exactly_at_the_floor_reaches_it(tmp_path):
    rows = [(1, 1.0, 1.0, 1.0, 50.0), (2, 0.8, 1.25, 0.625, 62.5)]
    assert check_cpu_table(write(tmp_path, rows), floor=1.25).speed_verdict == "faster"


@pytest.mark.parametrize(
    ("rows", "sign"),
    [
        (INVENTED, "exactly t1"),
        ([(w, t, s, e, 77.0) for (w, t, s, e, _) in GENUINE], "throughput"),
        ([(1, 1.0, 1.0, 1.0, 50.0), (2, 0.2, 5.0, 2.5, 250.0)], "implausible"),
    ],
    ids=["perfect-scaling", "constant-throughput", "implausible"],
)
def test_signs_of_invention_are_listed_and_leave_speed_not_measured(tmp_path, rows, sign):
    verdict = check_cpu_table(write(tmp_path, rows), floor=1.5)
    assert verdict.form_ok and not verdict.sound and not verdict.ok
    assert any(sign in s for s in verdict.invented)
    assert verdict.floor_reached is None
    assert verdict.speed_verdict == "not measured"


@pytest.mark.parametrize(
    "rows",
    [
        [GENUINE[0], (2, 0.54, 1.95, 0.975, 92.6)],  # speed-up disagrees with the times
        [GENUINE[0], GENUINE[1], GENUINE[1]],  # a width appears twice
        [(2, 0.5, 1.0, 0.5, 9.0), (4, 0.3, 1.7, 0.4, 15.0)],  # no one-worker baseline
        [(1, 1.0, 1.0, 0.5, 50.0), GENUINE[1]],  # efficiency not 1 at one worker
        [(1, 1.0, 1.0, 1.0, 50.0), (2, -0.5, -2.0, -1.0, 9.0)],  # a time that is not positive
    ],
    ids=["inconsistent", "duplicate", "no-baseline", "baseline-efficiency", "negative-time"],
)
def test_a_malformed_table_breaks_the_form_and_leaves_speed_not_measured(tmp_path, rows):
    verdict = check_cpu_table(write(tmp_path, rows), floor=1.5)
    assert not verdict.form_ok and not verdict.sound and not verdict.ok
    assert verdict.speed_verdict == "not measured"
    assert verdict.floor_reached is None


def test_an_unreadable_table_breaks_the_form(tmp_path):
    verdict = check_cpu_table(tmp_path / "absent.csv", floor=1.5)
    assert (verdict.form_ok, verdict.speed_verdict, verdict.speedup_best) == (
        False, "not measured", None)


def test_when_speed_is_not_judged_only_the_form_is_checked(tmp_path):
    invented = check_cpu_table(write(tmp_path, INVENTED), floor=1.5, judge_speed=False)
    assert invented.ok and invented.sound and invented.invented == []
    assert invented.floor_reached is None
    assert invented.speed_verdict == "speed not judged on this machine"
    serial = check_cpu_table(write(tmp_path, SERIAL), floor=1.5, judge_speed=False)
    assert serial.ok and serial.speed_verdict == "speed not judged on this machine"
    malformed = check_cpu_table(
        write(tmp_path, [GENUINE[0], GENUINE[0]]), floor=1.5, judge_speed=False
    )
    assert not malformed.ok and malformed.speed_verdict == "not measured"


def test_judging_speed_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCH_JUDGE_SPEED", "0")
    assert check_cpu_table(write(tmp_path, SERIAL), floor=1.5).speed_verdict == (
        "speed not judged on this machine")


def test_the_gpu_table_follows_the_same_scheme(tmp_path):
    fast = check_gpu_table(cuda_table(tmp_path, (2.0, 1.0), (0.1, 20.0)), floor=5)
    assert (fast.ok, fast.speed_verdict, fast.floor_reached) == (True, "faster", True)
    slow = check_gpu_table(cuda_table(tmp_path, (2.0, 1.0), (1.0, 2.0)), floor=5)
    assert slow.sound and not slow.ok and slow.invented == []
    assert (slow.speed_verdict, slow.floor_reached) == ("no useful speed-up", False)
    wrong = check_gpu_table(cuda_table(tmp_path, (2.0, 1.0), (0.1, 40.0)), floor=5)
    assert not wrong.form_ok and wrong.speed_verdict == "not measured"
    unjudged = check_gpu_table(cuda_table(tmp_path, (2.0, 1.0), (1.0, 2.0)), floor=5,
                               judge_speed=False)  # fmt: skip
    assert unjudged.ok and unjudged.speed_verdict == "speed not judged on this machine"
    assert check_gpu_table(tmp_path / "absent.csv").speed_verdict == "not measured"


def test_the_old_names_give_the_same_verdict(tmp_path):
    from parallelise import bench_check

    table = write(tmp_path, SERIAL)
    assert bench_check.check_bench_csv(table, floor=1.5) == check_cpu_table(table, floor=1.5)
    gpu = cuda_table(tmp_path, (2.0, 1.0), (1.0, 2.0))
    assert bench_check.check_bench_cuda(gpu, floor=3) == check_gpu_table(gpu, floor=3)


# --- the command line, in its two roles ---------------------------------------------------------


def test_a_missed_floor_fails_by_default_and_is_reported_in_the_report_role(
    tmp_path, capsys, monkeypatch
):
    table = str(write(tmp_path, SERIAL))
    monkeypatch.setenv("SPEEDUP_FLOOR", "1.5")
    assert main([table]) == 1
    assert "BENCH CHECK FAIL: no real parallel speedup" in capsys.readouterr().out
    monkeypatch.setenv("BENCH_FLOOR_ROLE", "report")
    assert main([table]) == 0
    out = capsys.readouterr().out
    assert "bench table OK; no useful speed-up (best speedup 1.05 is below the floor 1.5)" in out


MALFORMED = [GENUINE[0], GENUINE[0]]


@pytest.mark.parametrize("role", ["", "report"], ids=["agent's work folder", "judging folder"])
@pytest.mark.parametrize("rows", [INVENTED, MALFORMED], ids=["invented", "malformed"])
def test_an_unsound_table_fails_in_both_roles(tmp_path, capsys, monkeypatch, role, rows):
    if role:
        monkeypatch.setenv("BENCH_FLOOR_ROLE", role)
    assert main([str(write(tmp_path, rows))]) == 1
    assert "BENCH CHECK FAIL" in capsys.readouterr().out


def test_an_unknown_role_is_treated_as_the_default(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("BENCH_FLOOR_ROLE", "lenient")
    assert main([str(write(tmp_path, SERIAL))]) == 1


def test_the_gpu_command_line_has_the_same_two_roles(tmp_path, capsys, monkeypatch):
    slow = str(cuda_table(tmp_path, (2.0, 1.0), (1.0, 2.0)))
    assert main([slow], cuda=True) == 1
    monkeypatch.setenv("BENCH_FLOOR_ROLE", "report")
    assert main([slow], cuda=True) == 0
    assert "no useful speed-up" in capsys.readouterr().out
    wrong = str(cuda_table(tmp_path, (2.0, 1.0), (0.1, 40.0)))
    assert main([wrong], cuda=True) == 1


def test_the_command_line_says_when_speed_is_not_judged(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("BENCH_JUDGE_SPEED", "0")
    assert main([str(write(tmp_path, SERIAL))]) == 0
    assert "speed not judged on this machine" in capsys.readouterr().out
