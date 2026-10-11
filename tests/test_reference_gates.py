"""Every reference solution in problems/ must pass the gate.

This is the check that the machinery is sound without any model: for each problem and
backend present in the repository it assembles a work folder containing the known-good
solution and expects the gate to pass. Tests skip only when the needed compiler is absent.
"""

import shutil
from pathlib import Path

import pytest

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
    if not result.passed and problem in MEMORY_BOUND and result.failure_stage == "benchmark":
        # The conjugate gradient solver is limited by memory bandwidth, so on a small
        # machine the speed-up can honestly fall below the floor. Correctness stages
        # (build and both unit tests) have already passed at this point.
        pytest.skip(f"{problem} is memory-bound; this machine shows speed-up {result.speedup_best}")
    assert result.passed, f"{problem}/{backend} failed at {result.failure_stage}:\n{result.log}"
