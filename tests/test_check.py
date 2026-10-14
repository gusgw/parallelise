"""Tests for the check, using the synthetic OpenMP fixture and deliberately broken kernels."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from tests.small_machine import skip_if_only_speedup_failed

from parallelise.assemble import assemble_work_dir
from parallelise.check import FAILURE_STAGES, _tampered_files, run_check

FIXTURE = Path(__file__).parent / "fixtures" / "midpoint"


def have_openmp_compiler() -> bool:
    cc = shutil.which("gcc") or shutil.which("cc")
    if not cc or not shutil.which("make"):
        return False
    probe = subprocess.run(
        [cc, "-fopenmp", "-x", "c", "-", "-o", "/dev/null"],
        input="#include <omp.h>\nint main(void){return omp_get_max_threads()<0;}\n",
        capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    return probe.returncode == 0


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not have_openmp_compiler(), reason="needs gcc with OpenMP and make"),
]


def folder(tmp_path: Path, kernel_source: str | None = None) -> Path:
    work = assemble_work_dir(FIXTURE, "openmp", tmp_path / "w", include_reference=True)
    if kernel_source is not None:
        (work / "midpoint.c").write_text(kernel_source)
    return work


HEADER = '#include <math.h>\n#include <omp.h>\n#include "midpoint.h"\n'


def test_failure_stage_names_match_the_contract():
    assert set(FAILURE_STAGES) == {
        "none", "no-files", "tampered", "build", "test-one-worker", "test-many-workers",
        "benchmark", "timeout",
    }  # fmt: skip


def test_reference_solution_passes(tmp_path):
    work = folder(tmp_path)
    result = run_check(work, "openmp", timeout_s=240)
    skip_if_only_speedup_failed(result, work, "the fixture reference")
    assert result.passed, result.log
    assert result.failure_stage == "none"
    assert result.speedup_best and result.speedup_best > 1.3
    assert "bench stats OK" in result.log


def test_missing_kernel_is_no_files(tmp_path):
    work = folder(tmp_path)
    (work / "midpoint.c").unlink()
    result = run_check(work, "openmp")
    assert (result.passed, result.failure_stage) == (False, "no-files")


def test_warning_is_a_build_failure(tmp_path):
    source = HEADER + "double midpoint(double a, double b, long n)\n" \
        "{ int unused; return (b - a) * (double)n * 0.0; }\n"  # fmt: skip
    result = run_check(folder(tmp_path, source), "openmp")
    assert result.failure_stage == "build", result.log


def test_wrong_answer_fails_at_one_worker(tmp_path):
    source = HEADER + "double midpoint(double a, double b, long n)\n" \
        "{ (void)a; (void)b; (void)n; return 1.0; }\n"  # fmt: skip
    result = run_check(folder(tmp_path, source), "openmp")
    assert result.failure_stage == "test-one-worker", result.log


def test_answer_wrong_only_with_threads_fails_at_many_workers(tmp_path):
    reference = (FIXTURE / "openmp/reference/midpoint.c").read_text()
    broken = reference.replace(
        "return integral_sum * h;",
        "return omp_get_max_threads() > 1 ? 1.0 : integral_sum * h;",
    ).replace("#include <math.h>", "#include <math.h>\n#include <omp.h>")
    result = run_check(folder(tmp_path, broken), "openmp")
    assert result.failure_stage == "test-many-workers", result.log


def test_serial_kernel_fails_the_benchmark(tmp_path):
    reference = (FIXTURE / "openmp/reference/midpoint.c").read_text()
    serial = reference.replace("#pragma omp parallel for reduction(+ : integral_sum)\n", "")
    assert serial != reference
    result = run_check(folder(tmp_path, serial), "openmp", timeout_s=240)
    assert result.failure_stage == "benchmark", result.log
    assert "BENCH CHECK FAIL" in result.log


def test_stricter_floor_than_the_makefile_fails(tmp_path):
    result = run_check(folder(tmp_path), "openmp", timeout_s=240, speedup_floor=1000.0)
    assert result.failure_stage == "benchmark"
    assert not result.passed


def test_hanging_test_is_a_timeout(tmp_path):
    source = HEADER + "double midpoint(double a, double b, long n)\n" \
        "{ (void)a; (void)b; (void)n; for (;;) {} return 0.0; }\n"  # fmt: skip
    result = run_check(folder(tmp_path, source), "openmp", timeout_s=5)
    assert result.failure_stage == "timeout", result.log


def test_clean_folder_has_matching_digests(tmp_path):
    work = folder(tmp_path)
    manifest = json.loads((work / ".trusted.json").read_text())["sha256"]
    for name in ("Makefile", "bench.c", "testing.h", "midpoint.h", "check_bench.py"):
        assert name in manifest
    assert "midpoint.c" not in manifest
    assert _tampered_files(work) == []


@pytest.mark.parametrize("victim", ["bench.c", "Makefile", "testing.h", "midpoint.h"])
def test_edited_trusted_file_is_tampered(tmp_path, victim):
    work = folder(tmp_path)
    with (work / victim).open("a") as handle:
        handle.write("\n/* edited */\n")
    result = run_check(work, "openmp", timeout_s=60)
    assert (result.passed, result.failure_stage) == (False, "tampered")
    assert f"{victim} has been changed" in result.log


def test_removed_trusted_file_is_tampered(tmp_path):
    work = folder(tmp_path)
    (work / "bench_sweep.sh").unlink()
    assert run_check(work, "openmp", timeout_s=60).failure_stage == "tampered"


def test_missing_manifest_is_tampered(tmp_path):
    work = folder(tmp_path)
    (work / ".trusted.json").unlink()
    result = run_check(work, "openmp", timeout_s=60)
    assert result.failure_stage == "tampered"


def test_agent_tool_can_still_write_the_kernel(tmp_path):
    from parallelise.agents.tools import write_file

    work = folder(tmp_path)
    write_file(work, "midpoint.c", HEADER + "double midpoint(double a, double b, long n)\n"
              "{ (void)a; (void)b; (void)n; return 1.0; }\n")  # fmt: skip
    assert _tampered_files(work) == []
