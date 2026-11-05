"""Validation of benchmark tables: whether a table is sound, and what it says about speed.

In the first version of the harness, in July 2026, the agent ran the benchmark itself and
the harness read the table that the agent left behind. An agent can print a table of the
right shape containing made-up figures, and one routine did exactly that: it hard-coded
1.0, 2.0 and 4.0 and printed the time, half the time and a quarter of the time as if they
had been measured.

The fix was to make the benchmark program a trusted file that the agent cannot change, and
the check now runs it afresh at judgement time, overwriting any table the agent left.

The rules below remain as a second line of defence. They are needed, for example, when a
file named ``GNUmakefile`` makes ``make`` run the agent's own recipe instead of the trusted
one, so that the table might not come from the trusted benchmark after all.

The rules are of three kinds. For the worker-count sweep (OpenMP threads or MPI ranks) the
columns are the worker count, ``time_s``, ``speedup``, ``efficiency`` and optionally
``throughput``.

The rules of form say that the table can be trusted as a table:

- it can be read, has the right columns and at least two rows;
- each width appears once, since a genuine sweep measures each width once, and repeats
  suggest a re-run or an invented table;
- the first row is the one-worker baseline, with speed-up 1.0 and efficiency 1.0, which
  catches the common mistake of reporting one over the time instead of t1 over t;
- every time is positive;
- each reported speed-up equals t1 divided by that row's time, so that the figure cannot
  disagree with the times it is supposed to come from.

The signs of invention say that the figures look made up rather than measured:

- every parallel time is exactly t1 divided by the width, which real timings never are;
- the throughput, if given, does not change with the width;
- the best speed-up is above 1.5 times the widest worker count.

The floor is the least best speed-up that counts as a useful speed-up. A serial routine
repeated on every worker only wobbles near 1.0, and so does a routine that is limited by
memory bandwidth on a machine where one worker already uses all of it. Missing the floor
does not make a table unsound: it is a finding about speed.

A table is sound when it keeps every rule of form and shows no sign of invention. The
verdict then answers the question of speed in one of four ways: "faster" (the best speed-up
reached the floor), "no useful speed-up" (it did not), "speed not judged on this machine"
(speed is switched off, see ``BENCH_JUDGE_SPEED``) or "not measured" (the table is missing,
unsound or invented, so it says nothing about speed). When speed is not judged, only the
rules of form are applied, because on a shared or very small machine the timings say nothing.

The CUDA table has one row for the serial baseline and one for the graphics card, both timed
by the trusted benchmark. Its rules of form are that both rows are present, both times are
positive, the serial speed-up is 1.0 and the reported speed-up agrees with the times. It has
no worker-count sweep, so there is no sign of invention to look for, and its floor is higher.

The command-line entry, ``main``, is what the ``check_bench.py`` script in a work folder runs
from the Makefile's ``check`` target. It always fails a table that is unsound. What it does
when the floor is missed depends on ``BENCH_FLOOR_ROLE``: by default it fails, because the
agent's own ``make check`` is how the agent learns that it has not yet parallelised its
routine. The check sets ``BENCH_FLOOR_ROLE=report`` in its judging folder, and then a missed
floor is reported as "no useful speed-up" and the script succeeds, because the check judges
correctness and reports speed separately.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("parallelise.bench_check")

BASELINE_TOLERANCE = 0.05  # how far from 1.0 the one-worker speed-up may be
CONSISTENCY_TOLERANCE = 0.02  # reported speed-up must match t1/t within this fraction
EXACT_TOLERANCE = 1.0e-9  # relative closeness at which a time is "exactly" t1/width
THROUGHPUT_VARIATION = 1.05  # throughput must vary by at least this ratio
DEFAULT_FLOOR = 1.5
DEFAULT_CUDA_FLOOR = 3.0
PLAUSIBLE_FACTOR = 1.5  # speed-up may exceed the width by this factor at most

#: The best speed-up reached the floor.
FASTER = "faster"
#: The table is sound, but the best speed-up is below the floor.
NO_USEFUL_SPEEDUP = "no useful speed-up"
#: Speed is switched off on this machine, so only the form of the table was checked.
SPEED_NOT_JUDGED = "speed not judged on this machine"
#: There is no sound table to judge: it is missing, malformed or invented.
NOT_MEASURED = "not measured"
#: Every speed verdict, in the order in which reports list them.
SPEED_VERDICTS = (FASTER, NO_USEFUL_SPEEDUP, SPEED_NOT_JUDGED, NOT_MEASURED)

#: Set to 0 in the environment of a benchmark run to check the table's form but not its
#: timings. The check sets it when PARALLELISE_JUDGE_SPEED is 0, for machines such as shared
#: test servers whose timings mean nothing.
JUDGE_SPEED_VARIABLE = "BENCH_JUDGE_SPEED"

#: Says what ``main`` does with a missed floor. Unset, or any value but ``report``, means
#: that a missed floor fails, as the agent's own ``make check`` needs. The check sets it to
#: ``report`` in its judging folder, so that a missed floor is reported and does not fail.
FLOOR_ROLE_VARIABLE = "BENCH_FLOOR_ROLE"
#: The value of ``FLOOR_ROLE_VARIABLE`` under which a missed floor is reported, not failed.
REPORT_ROLE = "report"


@dataclass
class BenchVerdict:
    """The outcome of checking a benchmark table.

    ``form_ok`` is true when the table keeps every rule of form. ``invented`` lists the signs
    of invention that were found; it is empty when none were, and when speed is not judged,
    since they are then not looked for. ``floor`` is the floor that was applied, and
    ``floor_reached`` says whether the best speed-up reached it; it is None when the floor
    was not applied, because the table is unsound or speed is not judged. ``speedup_best``
    is the best speed-up in the table, or None when the table has no speed-up to read.
    ``speed_verdict`` is one of ``SPEED_VERDICTS``. ``reasons`` describes every broken rule
    and a missed floor, in that order. ``ok`` is true when the table is sound and the floor
    was reached or not applied, which is what the agent's own ``make check`` requires.
    """

    ok: bool
    reasons: list[str] = field(default_factory=list)
    speedup_best: float | None = None
    form_ok: bool = True
    invented: list[str] = field(default_factory=list)
    floor: float | None = None
    floor_reached: bool | None = None
    speed_verdict: str = NOT_MEASURED

    @property
    def sound(self) -> bool:
        """True when the table keeps every rule of form and shows no sign of invention."""
        return self.form_ok and not self.invented

    @property
    def message(self) -> str:
        """A one-paragraph description suitable for printing."""
        if self.ok:
            return "bench stats OK"
        return "BENCH CHECK FAIL: " + "; ".join(self.reasons)


def floor_from_env(default: float) -> float:
    """Read the speed-up floor from SPEEDUP_FLOOR, falling back to ``default``."""
    raw = os.environ.get("SPEEDUP_FLOOR", "")
    try:
        return float(raw) if raw.strip() else default
    except ValueError:
        log.warning("ignoring unreadable SPEEDUP_FLOOR=%r", raw)
        return default


def speed_judged_from_env() -> bool:
    """Return False when the benchmark's environment says that speed is not to be judged."""
    return os.environ.get(JUDGE_SPEED_VARIABLE, "1").strip() != "0"


def floor_reported_from_env() -> bool:
    """Return True when BENCH_FLOOR_ROLE is ``report``, so that a missed floor does not fail."""
    return os.environ.get(FLOOR_ROLE_VARIABLE, "").strip().lower() == REPORT_ROLE


def _verdict(
    form: list[str],
    invented: list[str],
    best: float | None,
    floor: float,
    judge_speed: bool,
    missed: Callable[[float], str],
) -> BenchVerdict:
    """Combine the findings about a table into a verdict.

    The floor is applied only to a sound table whose speed is judged; ``missed`` gives the
    reason to record when the best speed-up falls below it.
    """
    reasons = form + invented
    sound = not reasons
    floor_reached: bool | None = None
    if not sound or best is None:
        speed = NOT_MEASURED
    elif not judge_speed:
        speed = SPEED_NOT_JUDGED
    else:
        floor_reached = best >= floor
        speed = FASTER if floor_reached else NO_USEFUL_SPEEDUP
        if not floor_reached:
            reasons.append(missed(best))
    return BenchVerdict(
        ok=sound and floor_reached is not False,
        reasons=reasons,
        speedup_best=best,
        form_ok=not form,
        invented=list(invented),
        floor=floor,
        floor_reached=floor_reached,
        speed_verdict=speed,
    )


def _read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _parse_sweep(lines: list[str]) -> tuple[str, list[dict[str, float]], list[str]]:
    """Return the unit name, the numeric rows and any structural complaints."""
    header = next(
        (i for i, ln in enumerate(lines) if ln.lower().startswith(("threads", "ranks"))), None
    )
    if header is None:
        return "threads", [], ["no CSV header line starting with 'threads' or 'ranks'"]
    cols = [c.strip().lower() for c in lines[header].split(",")]
    unit = "threads" if "threads" in cols else "ranks"
    missing = [c for c in (unit, "time_s", "speedup", "efficiency") if c not in cols]
    if missing:
        return unit, [], ["missing column: " + ", ".join(missing)]
    rows: list[dict[str, float]] = []
    for ln in lines[header + 1 :]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) != len(cols):
            continue
        try:
            rows.append({c: float(v) for c, v in zip(cols, parts, strict=True)})
        except ValueError:
            continue
    if len(rows) < 2:
        return unit, rows, [f"need at least two rows, got {len(rows)}"]
    return unit, rows, []


def _sweep_form(unit: str, rows: list[dict[str, float]]) -> tuple[list[str], bool]:
    """Apply the rules of form to the parsed rows, which must be sorted by width.

    Returns the broken rules and whether the rows got far enough to be examined for signs of
    invention: a repeated width, a missing baseline or a time that is not positive stops the
    examination at once.
    """
    widths = [int(r[unit]) for r in rows]
    if len(set(widths)) != len(widths):
        return [
            f"duplicate {unit} values {widths}: a sweep should measure each width once "
            "(a re-run or an invented table)"
        ], False
    first = rows[0]
    if widths[0] != 1:
        return [f"first row must be 1 {unit[:-1]} (the baseline), got {widths[0]}"], False
    form: list[str] = []
    if abs(first["speedup"] - 1.0) > BASELINE_TOLERANCE:
        form.append(
            f"speedup at 1 {unit[:-1]} must be 1.0, got {first['speedup']:g} "
            "(was it computed as one over the time instead of t1 over t?)"
        )
    if abs(first["efficiency"] - 1.0) > BASELINE_TOLERANCE:
        form.append(f"efficiency at 1 {unit[:-1]} must be 1.0, got {first['efficiency']:g}")
    t1 = first["time_s"]
    if t1 <= 0:
        form.append(f"baseline time_s must be positive, got {t1:g}")
        return form, False
    for r in rows:
        w, t = int(r[unit]), r["time_s"]
        if t <= 0:
            form.append(f"time_s must be positive, got {t:g} at {w} {unit}")
            return form, False
        want = t1 / t
        if abs(r["speedup"] - want) > CONSISTENCY_TOLERANCE * max(want, 1.0):
            form.append(
                f"reported speedup {r['speedup']:g} at {w} {unit} disagrees with the times "
                f"(t1/t = {want:g}); the speedup column is not derived from the measurements"
            )
    return form, True


def _sweep_invention(unit: str, rows: list[dict[str, float]], best: float | None) -> list[str]:
    """Return the signs of invention in rows that are sorted by width and have positive times."""
    widths = [int(r[unit]) for r in rows]
    t1 = rows[0]["time_s"]
    signs: list[str] = []
    parallel = [r for r in rows if int(r[unit]) > 1]
    exact = [r for r in parallel if abs(r["time_s"] - t1 / int(r[unit])) <= EXACT_TOLERANCE * t1]
    if parallel and len(exact) == len(parallel):
        signs.append(
            "every parallel time is exactly t1 divided by the width: that is computed, "
            "not measured (invented perfect scaling)"
        )
    if "throughput" in rows[0]:
        tputs = [r["throughput"] for r in rows]
        lo, hi = min(tputs), max(tputs)
        if lo > 0 and hi / lo < THROUGHPUT_VARIATION:
            signs.append(
                f"throughput is essentially constant ({lo:g} to {hi:g}) across widths; it "
                "should change with the speed-up (invented)"
            )
    if best is not None and best > max(widths) * PLAUSIBLE_FACTOR:
        signs.append(f"implausible speedup {best:g} for {max(widths)} {unit}")
    return signs


def check_cpu_table(
    path: str | Path, floor: float | None = None, judge_speed: bool | None = None
) -> BenchVerdict:
    """Check a worker-count sweep table, and judge its speed.

    ``floor`` is the least best speed-up that counts as useful; when omitted it is read from
    the SPEEDUP_FLOOR environment variable, defaulting to 1.5. Every broken rule that can be
    found is returned, not only the first (see the module description for the rules).

    When ``judge_speed`` is False (by default, when BENCH_JUDGE_SPEED is 0), only the
    table's rules of form are applied, and the speed verdict of a sound table is "speed not
    judged on this machine". The signs of invention and the floor are not applied, because
    on a shared or very small machine they say nothing.
    """
    if floor is None:
        floor = floor_from_env(DEFAULT_FLOOR)
    if judge_speed is None:
        judge_speed = speed_judged_from_env()
    unit = "threads"

    def missed(best: float) -> str:
        return (
            f"no real parallel speedup (best {best:.2f}x is below {floor:g}x): the routine is "
            f"probably not decomposed across {unit} (serial work repeated on every worker)"
        )

    try:
        lines = _read_lines(Path(path))
    except (OSError, ValueError) as exc:  # ValueError: not text
        return _verdict([f"cannot read {path}: {exc}"], [], None, floor, judge_speed, missed)
    unit, rows, problems = _parse_sweep(lines)
    best = max((r["speedup"] for r in rows), default=None)
    if problems:
        return _verdict(problems, [], best, floor, judge_speed, missed)
    rows.sort(key=lambda r: r[unit])
    form, examinable = _sweep_form(unit, rows)
    invented = _sweep_invention(unit, rows, best) if examinable and judge_speed else []
    return _verdict(form, invented, best, floor, judge_speed, missed)


def sweep_rows(path: str | Path) -> list[tuple[int, float, float]]:
    """Return the width, time and speed-up of each row of a sweep table, sorted by width.

    It returns an empty list when the table cannot be read or has the wrong form, which
    ``check_cpu_table`` reports in its own words.
    """
    try:
        lines = _read_lines(Path(path))
    except (OSError, ValueError):
        return []
    unit, rows, problems = _parse_sweep(lines)
    if problems:
        return []
    return sorted((int(r[unit]), r["time_s"], r["speedup"]) for r in rows)


def check_gpu_table(
    path: str | Path, floor: float | None = None, judge_speed: bool | None = None
) -> BenchVerdict:
    """Check a CUDA table with columns ``impl,time_s,speedup`` and rows serial and gpu.

    CUDA has no worker-count sweep, so the evidence is the serial CPU run against the GPU
    run at the same size, both timed by the trusted benchmark. The rules of form are that
    both rows are present, both times are positive, the serial speed-up is 1.0 and the
    reported GPU speed-up agrees with the times. There are no signs of invention to look
    for. The floor, by default 3.0 or SPEEDUP_FLOOR when that is set, applies to the
    reported GPU speed-up, and ``judge_speed`` works as for ``check_cpu_table``.
    """
    if floor is None:
        floor = floor_from_env(DEFAULT_CUDA_FLOOR)
    if judge_speed is None:
        judge_speed = speed_judged_from_env()

    def missed(best: float) -> str:
        return (
            f"gpu speedup {best:.2f}x is below the floor {floor:g}x: the routine is probably "
            "not really running on the GPU"
        )

    try:
        lines = _read_lines(Path(path))
    except (OSError, ValueError) as exc:  # ValueError: not text
        return _verdict([f"cannot read {path}: {exc}"], [], None, floor, judge_speed, missed)
    rows: dict[str, tuple[float, float]] = {}
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or parts[0] == "impl":
            continue
        try:
            rows[parts[0]] = (float(parts[1]), float(parts[2]))
        except ValueError:
            continue
    missing = [n for n in ("serial", "gpu") if n not in rows]
    if missing:
        return _verdict(["missing row: " + ", ".join(missing)], [], None, floor, judge_speed,
                        missed)  # fmt: skip
    t_serial, su_serial = rows["serial"]
    t_gpu, su_gpu = rows["gpu"]
    if t_serial <= 0 or t_gpu <= 0:
        positive = f"times must be positive (serial {t_serial:g}, gpu {t_gpu:g})"
        return _verdict([positive], [], su_gpu, floor, judge_speed, missed)
    form: list[str] = []
    if abs(su_serial - 1.0) > 1e-9:
        form.append(f"serial speedup must be 1.0, got {su_serial:g}")
    want = t_serial / t_gpu
    if abs(su_gpu - want) > CONSISTENCY_TOLERANCE * max(want, 1.0):
        form.append(
            f"reported gpu speedup {su_gpu:g} disagrees with the times (t_serial/t_gpu = {want:g})"
        )
    return _verdict(form, [], su_gpu, floor, judge_speed, missed)


def check_bench_csv(path: str | Path, floor: float | None = None) -> BenchVerdict:
    """Deprecated name of ``check_cpu_table``, kept so that outside callers do not break."""
    return check_cpu_table(path, floor)


def check_bench_cuda(path: str | Path, floor: float | None = None) -> BenchVerdict:
    """Deprecated name of ``check_gpu_table``, kept so that outside callers do not break."""
    return check_gpu_table(path, floor)


def main(argv: Sequence[str] | None = None, cuda: bool = False) -> int:
    """Command-line entry: check the CSV named by the first argument, and return the status.

    The floor comes from the SPEEDUP_FLOOR environment variable. A table that cannot be
    read, breaks a rule of form or shows a sign of invention is always reported with its
    reasons and gives 1. A sound table gives 0 when the floor was reached (``bench stats
    OK``) or speed is not judged. When the floor was missed, the result depends on
    BENCH_FLOOR_ROLE: by default the reasons are printed and the result is 1, so that the
    agent's own ``make check`` fails until the routine is parallel; with ``report`` the
    verdict "no useful speed-up" is printed and the result is 0. A wrong number of
    arguments gives 2.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: check_bench.py <bench.csv>")
        return 2
    verdict = (check_gpu_table if cuda else check_cpu_table)(args[0])
    if not verdict.sound:
        print(verdict.message)
        return 1
    if verdict.speed_verdict == SPEED_NOT_JUDGED:
        print("bench table OK; speed not judged on this machine")
        return 0
    if verdict.floor_reached:
        best = f" (best speedup {verdict.speedup_best:.2f})" if verdict.speedup_best else ""
        print("bench stats OK" + best)
        return 0
    if verdict.speed_verdict == NO_USEFUL_SPEEDUP and floor_reported_from_env():
        print(
            f"bench table OK; {NO_USEFUL_SPEEDUP} (best speedup {verdict.speedup_best:.2f} "
            f"is below the floor {verdict.floor:g})"
        )
        return 0
    print(verdict.message)
    return 1


if __name__ == "__main__":
    sys.exit(main())
