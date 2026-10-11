"""Every reference solution in problems/ must pass the gate.

This is the check that the machinery is sound without any model: for each problem and
backend present in the repository it assembles a work folder containing the known-good
solution and expects the gate to pass. Tests skip only when the needed compiler is absent.
"""

import os
import shutil
from pathlib import Path

import pytest

from parallelise.assemble import assemble_work_dir
from parallelise.bench_check import check_bench_csv
from parallelise.gate import run_gate

PROBLEMS = Path(__file__).resolve().parent.parent / "problems"

# Problems whose speed-up depends strongly on the memory system of the test machine.
MEMORY_BOUND = {"cg"}

# Hosted continuous-integration runners have two to four shared cores, so a genuinely
# parallel kernel may show no speed-up there. When this variable is set, a reference
# solution that passes every correctness stage and fails only the speed-up floor is
# reported as skipped rather than failed; the speed-up requirement is then checked on a
# multi-core host instead.
SPEEDUP_UNRELIABLE = os.environ.get("PARALLELISE_SPEEDUP_UNRELIABLE") == "1"


def cases() -> list:
    found = []
    for backend_dir in sorted(PROBLEMS.glob("*/*/")):
        problem, backend = backend_dir.parent.name, backend_dir.name
        if backend not in ("openmp", "mpi", "cuda") or not (backend_dir / "reference").is_dir():
            continue
        marks = [pytest.mark.slow]
        if backend in ("mpi", "cuda"):
            marks.append(getattr(pytest.mark, backend))
        found.append(pytest.param(problem, backend, marks=marks, id=f"{problem}-{backend}"))
    return found


def missing_tool(backend: str) -> str | None:
    needed = {"openmp": ["make", "gcc"], "mpi": ["make", "mpicc", "mpirun"],
              "cuda": ["make", "nvcc"]}[backend]  # fmt: skip
    return next((t for t in needed if shutil.which(t) is None), None)


@pytest.mark.parametrize(("problem", "backend"), cases())
def test_reference_solution_passes_gate(problem, backend, tmp_path):
    tool = missing_tool(backend)
    if tool:
        pytest.skip(f"{tool} is not installed")
    work = assemble_work_dir(PROBLEMS / problem, backend, tmp_path / "w", include_reference=True)
    result = run_gate(work, backend, timeout_s=600)
    if not result.passed and result.failure_stage == "benchmark":
        # Speed-up figures vary with whatever else the machine is doing, so a missed
        # floor gets one second attempt. A genuinely slow solution fails both times.
        result = run_gate(work, backend, timeout_s=600)
    if (
        not result.passed
        and result.failure_stage == "benchmark"
        and (SPEEDUP_UNRELIABLE or problem in MEMORY_BOUND)
    ):
        # Build and both unit tests have passed by this point. Re-check the benchmark
        # table with no floor: if it is otherwise sound, the only complaint was the
        # speed-up, which this machine cannot be expected to show.
        bench_csv = work / "bench.csv"
        if bench_csv.is_file() and check_bench_csv(bench_csv, floor=0.0).ok:
            pytest.skip(
                f"{problem}/{backend} is correct but shows speed-up {result.speedup_best} "
                f"on this {os.cpu_count()}-core machine; the speed-up requirement is "
                "checked on a multi-core host"
            )
    assert result.passed, f"{problem}/{backend} failed at {result.failure_stage}:\n{result.log}"
