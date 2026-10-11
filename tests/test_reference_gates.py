"""Every reference solution in problems/ must pass the gate.

This is the check that the machinery is sound without any model: for each problem and
backend present in the repository it assembles a work folder containing the known-good
solution and expects the gate to pass. Tests skip only when the needed compiler is absent.
"""

import shutil
from pathlib import Path

import pytest
from tests.small_machine import skip_if_only_speedup_failed

from parallelise.assemble import assemble_work_dir
from parallelise.gate import run_gate

PROBLEMS = Path(__file__).resolve().parent.parent / "problems"

# Problems whose speed-up depends strongly on the memory system of the test machine.
MEMORY_BOUND = {"cg"}



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
    skip_if_only_speedup_failed(
        result, work, f"{problem}/{backend}", always=problem in MEMORY_BOUND
    )
    assert result.passed, f"{problem}/{backend} failed at {result.failure_stage}:\n{result.log}"
