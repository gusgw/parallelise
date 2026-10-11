"""Validation of benchmark tables, so that invented speed-ups are rejected.

An agent can easily print a table of the right shape containing made-up figures; one
kernel was seen to hard-code 1.0, 2.0 and 4.0 and print the time, half the time and a
quarter of the time. The rules below therefore require the table both to be shaped
correctly and to look measured.

For the worker-count sweep (OpenMP threads or MPI ranks) the columns are the worker
count, ``time_s``, ``speedup``, ``efficiency`` and optionally ``throughput``:

- baseline: a one-worker row exists, with speed-up 1.0 and efficiency 1.0. This catches
  the common mistake of reporting one over the time instead of t1 over t.
- unique: each width appears once. A genuine sweep measures each width once, so repeats
  suggest a re-run or an invented table.
- consistent: each reported speed-up equals t1 divided by that row's time, so the figure
  cannot disagree with the times it is supposed to come from.
- measured: the parallel times are not all exactly t1 divided by the width, which real
  timings never are, and the throughput, if given, changes with the width.
- range: the best speed-up reaches the floor (a serial kernel repeated on every worker
  only wobbles near 1.0) and does not exceed 1.5 times the widest worker count.

The CUDA table has one row for the serial baseline and one for the GPU, both timed by the
trusted benchmark, and the GPU speed-up must reach a higher floor.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("parallelise.bench_check")

BASELINE_TOLERANCE = 0.05  # how far from 1.0 the one-worker speed-up may be
CONSISTENCY_TOLERANCE = 0.02  # reported speed-up must match t1/t within this fraction
EXACT_TOLERANCE = 1.0e-9  # relative closeness at which a time is "exactly" t1/width
THROUGHPUT_VARIATION = 1.05  # throughput must vary by at least this ratio
DEFAULT_FLOOR = 1.5
DEFAULT_CUDA_FLOOR = 5.0
PLAUSIBLE_FACTOR = 1.5  # speed-up may exceed the width by this factor at most


@dataclass
class BenchVerdict:
    """The outcome of checking a benchmark table."""

    ok: bool
    reasons: list[str] = field(default_factory=list)
    speedup_best: float | None = None

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


def check_bench_csv(path: str | Path, floor: float | None = None) -> BenchVerdict:
    """Check a worker-count sweep table.

    ``floor`` is the least acceptable best speed-up; when omitted it is read from the
    SPEEDUP_FLOOR environment variable, defaulting to 1.5. All reasons for rejection that
    can be found are returned, not only the first.
    """
    if floor is None:
        floor = floor_from_env(DEFAULT_FLOOR)
    try:
        lines = _read_lines(Path(path))
    except OSError as exc:
        return BenchVerdict(False, [f"cannot read {path}: {exc}"])
    unit, rows, problems = _parse_sweep(lines)
    best = max((r["speedup"] for r in rows), default=None)
    if problems:
        return BenchVerdict(False, problems, best)

    rows.sort(key=lambda r: r[unit])
    widths = [int(r[unit]) for r in rows]
    reasons: list[str] = []

    if len(set(widths)) != len(widths):
        reasons.append(
            f"duplicate {unit} values {widths}: a sweep should measure each width once "
            "(a re-run or an invented table)"
        )
        return BenchVerdict(False, reasons, best)

    first = rows[0]
    if widths[0] != 1:
        reasons.append(f"first row must be 1 {unit[:-1]} (the baseline), got {widths[0]}")
        return BenchVerdict(False, reasons, best)
    if abs(first["speedup"] - 1.0) > BASELINE_TOLERANCE:
        reasons.append(
            f"speedup at 1 {unit[:-1]} must be 1.0, got {first['speedup']:g} "
            "(was it computed as one over the time instead of t1 over t?)"
        )
    if abs(first["efficiency"] - 1.0) > BASELINE_TOLERANCE:
        reasons.append(f"efficiency at 1 {unit[:-1]} must be 1.0, got {first['efficiency']:g}")
    t1 = first["time_s"]
    if t1 <= 0:
        reasons.append(f"baseline time_s must be positive, got {t1:g}")
        return BenchVerdict(False, reasons, best)

    exact_hits = 0
    for r in rows:
        w = int(r[unit])
        t = r["time_s"]
        if t <= 0:
            reasons.append(f"time_s must be positive, got {t:g} at {w} {unit}")
            return BenchVerdict(False, reasons, best)
        want = t1 / t
        if abs(r["speedup"] - want) > CONSISTENCY_TOLERANCE * max(want, 1.0):
            reasons.append(
                f"reported speedup {r['speedup']:g} at {w} {unit} disagrees with the times "
                f"(t1/t = {want:g}); the speedup column is not derived from the measurements"
            )
        if w > 1 and abs(t - t1 / w) <= EXACT_TOLERANCE * t1:
            exact_hits += 1
    parallel = sum(1 for w in widths if w > 1)
    if parallel and exact_hits == parallel:
        reasons.append(
            "every parallel time is exactly t1 divided by the width: that is computed, "
            "not measured (invented perfect scaling)"
        )

    if "throughput" in rows[0]:
        tputs = [r["throughput"] for r in rows]
        lo, hi = min(tputs), max(tputs)
        if lo > 0 and hi / lo < THROUGHPUT_VARIATION:
            reasons.append(
                f"throughput is essentially constant ({lo:g} to {hi:g}) across widths; it "
                "should change with the speed-up (invented)"
            )

    if best is not None and best < floor:
        reasons.append(
            f"no real parallel speedup (best {best:.2f}x is below {floor:g}x): the kernel is "
            f"probably not decomposed across {unit} (serial work repeated on every worker)"
        )
    if best is not None and best > max(widths) * PLAUSIBLE_FACTOR:
        reasons.append(f"implausible speedup {best:g} for {max(widths)} {unit}")

    return BenchVerdict(not reasons, reasons, best)


def check_bench_cuda(path: str | Path, floor: float | None = None) -> BenchVerdict:
    """Check a CUDA table with columns ``impl,time_s,speedup`` and rows serial and gpu.

    CUDA has no worker-count sweep, so the evidence is the serial CPU run against the GPU
    run at the same size, both timed by the trusted benchmark. The default floor is 5.0.
    """
    if floor is None:
        floor = floor_from_env(DEFAULT_CUDA_FLOOR)
    try:
        lines = _read_lines(Path(path))
    except OSError as exc:
        return BenchVerdict(False, [f"cannot read {path}: {exc}"])
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
        return BenchVerdict(False, ["missing row: " + ", ".join(missing)])
    t_serial, su_serial = rows["serial"]
    t_gpu, su_gpu = rows["gpu"]
    if t_serial <= 0 or t_gpu <= 0:
        return BenchVerdict(False, [f"times must be positive (serial {t_serial:g}, gpu {t_gpu:g})"])
    reasons: list[str] = []
    if abs(su_serial - 1.0) > 1e-9:
        reasons.append(f"serial speedup must be 1.0, got {su_serial:g}")
    want = t_serial / t_gpu
    if abs(su_gpu - want) > CONSISTENCY_TOLERANCE * max(want, 1.0):
        reasons.append(
            f"reported gpu speedup {su_gpu:g} disagrees with the times (t_serial/t_gpu = {want:g})"
        )
    if su_gpu < floor:
        reasons.append(
            f"gpu speedup {su_gpu:.2f}x is below the floor {floor:g}x: the kernel is probably "
            "not really running on the GPU"
        )
    return BenchVerdict(not reasons, reasons, su_gpu)


def main(argv: Sequence[str] | None = None, cuda: bool = False) -> int:
    """Command-line entry: check the CSV named by the first argument.

    Prints ``bench stats OK`` and returns 0, or prints the reasons and returns 1. The
    floor comes from the SPEEDUP_FLOOR environment variable.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: check_bench.py <bench.csv>")
        return 2
    verdict = (check_bench_cuda if cuda else check_bench_csv)(args[0])
    if verdict.ok:
        best = f" (best speedup {verdict.speedup_best:.2f})" if verdict.speedup_best else ""
        print("bench stats OK" + best)
        return 0
    print(verdict.message)
    return 1


if __name__ == "__main__":
    sys.exit(main())
