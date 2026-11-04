"""Tests for assembling a work folder from the synthetic fixture problem."""

import pytest

from parallelise.assemble import (
    AssembleError,
    assemble_work_dir,
    trusted_digest,
    unit_test_filename,
)
from parallelise.errors import ParalleliseError

FIXTURE = __import__("pathlib").Path(__file__).parent / "fixtures" / "midpoint"


def test_work_folder_without_reference(tmp_path):
    dest = assemble_work_dir(FIXTURE, "openmp", tmp_path / "w")
    names = {p.name for p in dest.iterdir()}
    assert names == {
        "testing.h", "bench_sweep.sh", "check_bench.py", "check_bench_cuda.py",
        "midpoint.ipynb", "reference.h", "midpoint.h", "bench.c", "Makefile", ".trusted.json",
    }  # fmt: skip
    assert (dest / "bench_sweep.sh").stat().st_mode & 0o100


def test_work_folder_with_reference(tmp_path):
    dest = assemble_work_dir(FIXTURE, "openmp", tmp_path / "w", include_reference=True)
    assert (dest / "midpoint.c").is_file()
    assert (dest / "test_midpoint.c").is_file()


def test_existing_destination_is_emptied(tmp_path):
    dest = tmp_path / "w"
    dest.mkdir()
    (dest / "stale.txt").write_text("old")
    assemble_work_dir(FIXTURE, "openmp", dest)
    assert not (dest / "stale.txt").exists()


def test_shim_runs_the_package_checker(tmp_path):
    import subprocess
    import sys

    dest = assemble_work_dir(FIXTURE, "openmp", tmp_path / "w")
    csv = dest / "bench.csv"
    csv.write_text("threads,time_s,speedup,efficiency\n1,1,1,1\n2,0.6,1.667,0.833\n")
    done = subprocess.run(
        [sys.executable, "check_bench.py", "bench.csv"],
        cwd=dest, capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr
    assert "bench stats OK" in done.stdout


def test_unknown_backend_and_missing_backend(tmp_path):
    with pytest.raises(AssembleError):
        assemble_work_dir(FIXTURE, "fortran", tmp_path / "w")
    with pytest.raises(AssembleError):
        assemble_work_dir(FIXTURE, "mpi", tmp_path / "w")


def test_folder_without_notebook(tmp_path):
    with pytest.raises(AssembleError):
        assemble_work_dir(tmp_path, "openmp", tmp_path / "w")


def test_assemble_error_is_a_plain_package_error(tmp_path):
    with pytest.raises(ParalleliseError):
        assemble_work_dir(FIXTURE, "fortran", tmp_path / "w")


def test_unit_test_is_plain_c_for_every_backend():
    assert unit_test_filename("integrate") == "test_integrate.c"


def test_trusted_digest_ignores_only_the_package_location_in_the_checker_scripts(tmp_path):
    first = assemble_work_dir(FIXTURE, "openmp", tmp_path / "a")
    second = assemble_work_dir(FIXTURE, "openmp", tmp_path / "b")
    script = second / "check_bench.py"
    lines = script.read_text().splitlines()
    script.write_text(
        "\n".join(
            "sys.path.insert(0, '/another/computer/src')" if ln.startswith("sys.path.insert")
            else ln
            for ln in lines
        ) + "\n"
    )  # fmt: skip
    assert script.read_text() != (first / "check_bench.py").read_text()
    assert trusted_digest(script) == trusted_digest(first / "check_bench.py")
    script.write_text(script.read_text() + "print('changed')\n")
    assert trusted_digest(script) != trusted_digest(first / "check_bench.py")
    # Every other file is compared byte for byte.
    (second / "bench.c").write_text((first / "bench.c").read_text() + " ")
    assert trusted_digest(second / "bench.c") != trusted_digest(first / "bench.c")
