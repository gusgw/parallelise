"""Every known-good routine must pass the judgement against its problem's serial routine.

The known-good routines are kept under ``tests/fixtures/known_good/<problem>/<backend>``. A
problem is tested once it has a serial routine; a backend whose compiler is missing here is
skipped and listed, so that ``pytest -rs`` shows what was not run.
"""

import shutil
from pathlib import Path

import pytest

from parallelise import judge, notebook

ROOT = Path(__file__).resolve().parent.parent
KNOWN_GOOD = Path(__file__).parent / "fixtures" / "known_good"
COMPILERS = {"openmp": "gcc", "mpi": "mpicc", "cuda": "nvcc"}

CASES = sorted(
    (routine.parent.name, routine.stem, routine)
    for routine in KNOWN_GOOD.glob("*/*.c*")
    if (ROOT / "problems" / routine.parent.name / "serial.c").is_file()
)


@pytest.mark.slow
@pytest.mark.parametrize(("problem", "backend", "routine"), CASES,
                         ids=[f"{p}-{b}" for p, b, _ in CASES])  # fmt: skip
def test_the_known_good_routine_is_correct(problem, backend, routine):
    if shutil.which(COMPILERS[backend]) is None:
        pytest.skip(f"{COMPILERS[backend]} is not installed here")
    if backend == "cuda" and shutil.which("nvidia-smi") is None:
        pytest.skip("there is no graphics card here")
    folder = ROOT / "problems" / problem
    nb = folder / f"{problem}.ipynb"
    verdict = judge.judge(nb, notebook.declaration(nb), routine, folder / "serial.c", backend)
    assert verdict.correct, verdict.log
