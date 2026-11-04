"""Every reference solution in problems/ must pass the check.

This is the check that the machinery is sound without any model: for each problem and
backend present in the repository it assembles a work folder containing the known-good
solution and expects the check, given the problem folder, to pass. Tests skip only when the
needed compiler is absent.
"""

import shutil
from pathlib import Path

import pytest

from parallelise.assemble import assemble_work_dir
from parallelise.check import run_check

PROBLEMS = Path(__file__).resolve().parent.parent / "problems"




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
def test_reference_solution_passes_check(problem, backend, tmp_path):
    tool = missing_tool(backend)
    if tool:
        pytest.skip(f"{tool} is not installed")
    problem_dir = PROBLEMS / problem
    work = assemble_work_dir(problem_dir, backend, tmp_path / "w", include_reference=True)
    result = run_check(work, problem_dir, backend, timeout_s=600)
    assert result.passed, f"{problem}/{backend} failed at {result.failure_stage}:\n{result.log}"
