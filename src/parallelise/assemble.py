"""Building a work folder for one problem and one backend.

The work folder is flat and self-contained. It holds everything an agent is allowed to
read and the trusted files the gate depends on: the shared test header and benchmark
script from the package, a notebook, the oracle in C form, the kernel header with the
signature to implement, the benchmark the agent cannot change, and the Makefile.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import stat
from pathlib import Path

log = logging.getLogger("parallelise.assemble")

BACKENDS = ("openmp", "mpi", "cuda")
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_FILES = ("testing.h", "bench_sweep.sh")
PROBLEM_LEVEL_FILES = ("reference.h", "reference.json", "problem.h")

#: Name of the file in the work folder that records the digests of the trusted files.
TRUSTED_MANIFEST = ".trusted.json"
#: Files the agent must not change. The gate checks them against ``TRUSTED_MANIFEST``.
#: The kernel header ``<name>.h`` is added to this list for each problem.
TRUSTED_FILES = (
    "bench.c", "Makefile", "testing.h", "bench_sweep.sh", "check_bench.py",
    "check_bench_cuda.py", "reference.h", "problem.h",
)  # fmt: skip

_SHIM = '''"""Calls the benchmark checker in the parallelise package, so there is one copy of it."""
import sys

sys.path.insert(0, {package_parent!r})
from parallelise.bench_check import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(cuda={cuda}))
'''


class AssembleError(Exception):
    """Raised when a problem folder is missing something the work folder needs."""


def kernel_name(problem_dir: Path) -> str:
    """Return the kernel name of a problem folder: the stem of its notebook."""
    notebooks = sorted(problem_dir.glob("*.ipynb"))
    if len(notebooks) != 1:
        raise AssembleError(
            f"{problem_dir} must contain exactly one notebook, found {len(notebooks)}"
        )
    return notebooks[0].stem


def kernel_filename(name: str, backend: str) -> str:
    """Return the name of the kernel source file: .cu for CUDA, .c otherwise."""
    return f"{name}.cu" if backend == "cuda" else f"{name}.c"


def _write_shims(dest: Path) -> None:
    """Write the two small checker scripts that the Makefiles call from the work folder."""
    package_parent = str(Path(__file__).resolve().parent.parent)
    for filename, cuda in (("check_bench.py", False), ("check_bench_cuda.py", True)):
        (dest / filename).write_text(
            _SHIM.format(package_parent=package_parent, cuda=cuda), encoding="utf-8"
        )


def file_digest(path: Path) -> str:
    """Return the SHA-256 digest of a file as hexadecimal text."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trusted_file_names(name: str) -> tuple[str, ...]:
    """Return the names of the trusted files for a kernel called ``name``."""
    return (*TRUSTED_FILES, f"{name}.h")


def _write_trusted_manifest(dest: Path, name: str) -> None:
    """Record the digest of every trusted file that is present in the work folder."""
    digests = {
        filename: file_digest(dest / filename)
        for filename in trusted_file_names(name)
        if (dest / filename).is_file()
    }
    (dest / TRUSTED_MANIFEST).write_text(
        json.dumps({"sha256": digests}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def assemble_work_dir(
    problem_dir: Path, backend: str, dest: Path, include_reference: bool = False
) -> Path:
    """Build a flat work folder at ``dest`` and return it.

    ``dest`` is emptied first if it already exists. With ``include_reference`` the
    known-good kernel and unit test are copied in too, which is how the machinery is tested
    without a model. Raises ``AssembleError`` if a required file is missing.
    """
    problem_dir, dest = Path(problem_dir), Path(dest)
    if backend not in BACKENDS:
        raise AssembleError(f"unknown backend {backend!r}; choose one of {', '.join(BACKENDS)}")
    backend_dir = problem_dir / backend
    if not backend_dir.is_dir():
        raise AssembleError(f"{problem_dir.name} has no {backend} backend ({backend_dir})")
    name = kernel_name(problem_dir)

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for filename in DATA_FILES:
        shutil.copy2(DATA_DIR / filename, dest / filename)
    (dest / "bench_sweep.sh").chmod((dest / "bench_sweep.sh").stat().st_mode | stat.S_IXUSR)
    _write_shims(dest)

    shutil.copy2(problem_dir / f"{name}.ipynb", dest / f"{name}.ipynb")
    for filename in PROBLEM_LEVEL_FILES:
        if (problem_dir / filename).is_file():
            shutil.copy2(problem_dir / filename, dest / filename)
    for filename in (f"{name}.h", "bench.c", "Makefile"):
        source = backend_dir / filename
        if not source.is_file():
            raise AssembleError(f"missing {source}")
        shutil.copy2(source, dest / filename)

    if include_reference:
        reference = backend_dir / "reference"
        for filename in (kernel_filename(name, backend), f"test_{name}.c"):
            source = reference / filename
            if not source.is_file():
                raise AssembleError(f"missing reference file {source}")
            shutil.copy2(source, dest / filename)
    _write_trusted_manifest(dest, name)
    log.info("assembled %s/%s into %s", name, backend, dest)
    return dest
