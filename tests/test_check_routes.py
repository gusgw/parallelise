"""Tests for the routes by which a routine fooled the check, and for judging a work folder as it
was assembled.

Each route is first shown to work with the call recorder switched off, which is how the check
behaved before the recorder existed, and then shown to be closed with it on. The routines are
versions, for the test problem, of the ones that fooled the check on the integrate problem.
"""

import json
import shutil
from pathlib import Path

import pytest
from tests.small_machine import needs_speed
from tests.test_check import have_openmp_compiler

from parallelise import check as check_module
from parallelise import cli
from parallelise.assemble import TRUSTED_MANIFEST, assemble_work_dir
from parallelise.check import run_check
from parallelise.errors import CheckError

FIXTURE = Path(__file__).parent / "fixtures" / "midpoint"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not have_openmp_compiler(), reason="needs gcc with OpenMP and make"),
]

# Right at the test's sizes (at most 100000 subintervals) and wrong above a million, which
# includes the benchmark's twenty million: there it sums only the first tenth of the range.
WRONG_AT_THE_BENCHMARK_SIZE = r"""#include <math.h>
#include "midpoint.h"

double midpoint(double a, double b, long n)
{
    long m = (n > 1000000L) ? n / 10 : n;
    double h = (b - a) / (double)n, s = 0.0;
#pragma omp parallel for reduction(+ : s)
    for (long i = 0; i < m; i++) {
        s += sin(a + ((double)i + 0.5) * h);
    }
    return s * h;
}
"""

# Wholly serial, but it repeats its sum 4/w times at w threads, so one thread looks slow.
SERIAL_AND_SLOW_AT_ONE_THREAD = r"""#include <math.h>
#include <omp.h>
#include "midpoint.h"

static double serial_sum(double a, double b, long n)
{
    double h = (b - a) / (double)n, s = 0.0;
    for (long i = 0; i < n; i++) {
        s += sin(a + ((double)i + 0.5) * h);
    }
    return s * h;
}

double midpoint(double a, double b, long n)
{
    int w = omp_get_max_threads();
    int reps = (w >= 4) ? 1 : 4 / w;
    volatile double r = 0.0;
    for (int k = 0; k < reps; k++) {
        r = serial_sum(a, b, n);
    }
    return r;
}
"""

# Serial, and keeps its answer between calls: a later call at the same size waits for the
# stored time divided by about 0.7 times the thread count, and returns the stored answer.
SERIAL_WITH_A_STORED_ANSWER = r"""#include <math.h>
#include <omp.h>
#include "midpoint.h"

static long last_n = -1;
static double last_r, last_t;

double midpoint(double a, double b, long n)
{
    int w = omp_get_max_threads();
    if (n == last_n) {
        double until = omp_get_wtime() + last_t / (0.7 * w + 0.3);
        while (omp_get_wtime() < until) {
        }
        return last_r;
    }
    double start = omp_get_wtime();
    double h = (b - a) / (double)n, s = 0.0;
    for (long i = 0; i < n; i++) {
        s += sin(a + ((double)i + 0.5) * h);
    }
    last_n = n;
    last_r = s * h;
    last_t = omp_get_wtime() - start;
    return last_r;
}
"""


def folder(tmp_path: Path, source: str | None = None, problem: Path = FIXTURE) -> Path:
    work = assemble_work_dir(problem, "openmp", tmp_path / "w", include_reference=True)
    if source is not None:
        (work / "midpoint.c").write_text(source)
    return work


@pytest.fixture
def recorder_off(monkeypatch):
    """Judge as the check did before the call recorder existed."""
    monkeypatch.setattr(check_module, "_prepare_recorder", lambda *args, **kwargs: None)


def test_wrong_answer_at_the_benchmark_size_passed_without_the_recorder(tmp_path, recorder_off):
    result = run_check(folder(tmp_path, WRONG_AT_THE_BENCHMARK_SIZE), FIXTURE, "openmp")
    assert (result.passed, result.failure_stage) == (True, "none"), result.log


def test_wrong_answer_at_the_benchmark_size_fails(tmp_path):
    work = folder(tmp_path, WRONG_AT_THE_BENCHMARK_SIZE)
    result = run_check(work, FIXTURE, "openmp")
    assert (result.passed, result.failure_stage) == (False, "benchmark"), result.log
    assert "At the benchmark's own size, the routine's answer is not the known-good" in result.log
    assert "differs from the known-good routine by" in result.log
    assert result.speed_verdict == "not measured"


@needs_speed
@pytest.mark.parametrize(
    "source", [SERIAL_AND_SLOW_AT_ONE_THREAD, SERIAL_WITH_A_STORED_ANSWER],
    ids=["slow-at-one-thread", "stored-answer"],
)  # fmt: skip
def test_serial_routine_was_reported_faster_without_the_recorder(tmp_path, recorder_off, source):
    result = run_check(folder(tmp_path, source), FIXTURE, "openmp")
    assert (result.passed, result.speed_verdict) == (True, "faster"), result.log


@needs_speed
@pytest.mark.parametrize(
    "source", [SERIAL_AND_SLOW_AT_ONE_THREAD, SERIAL_WITH_A_STORED_ANSWER],
    ids=["slow-at-one-thread", "stored-answer"],
)  # fmt: skip
def test_serial_routine_is_correct_but_not_faster(tmp_path, source):
    result = run_check(folder(tmp_path, source), FIXTURE, "openmp")
    assert (result.passed, result.failure_stage) == (True, "none"), result.log
    assert result.speed_verdict == "no useful speed-up", result.log
    assert result.speedup_best is not None and result.speedup_best < 1.3
    assert "processors busy on average, so it is credited with" in result.log


def test_known_good_routine_agrees_and_is_not_held_back(tmp_path):
    result = run_check(folder(tmp_path), FIXTURE, "openmp")
    assert result.passed, result.log
    assert "answers agree with the known-good routine's" in result.log


# --- judging a work folder as it was assembled ------------------------------------------------

WARM_UP = (
    "    /* One untimed warm-up run, so that one-time costs are paid before the timed run\n"
    "     * begins, as in the shipped benchmarks. */\n"
    "    (void)midpoint(0.0, 3.141592653589793, n);\n"
)


def problem_copy(tmp_path: Path, old_style: bool = False) -> Path:
    """Copy the test problem; the old style names the Makefile variable KERNEL and has no
    warm-up call in the benchmark, as work folders assembled before November 2026 do."""
    problem = tmp_path / "problems" / "midpoint"
    shutil.copytree(FIXTURE, problem)
    if old_style:
        make_old_style(problem / "openmp")
    return problem


def make_old_style(backend_dir: Path) -> None:
    makefile = backend_dir / "Makefile"
    makefile.write_text(makefile.read_text().replace("ROUTINE", "KERNEL"))
    bench = backend_dir / "bench.c"
    assert WARM_UP in bench.read_text()
    bench.write_text(bench.read_text().replace(WARM_UP, ""))


def test_a_changed_benchmark_is_tampered_but_can_be_judged_as_assembled(tmp_path):
    problem = problem_copy(tmp_path)
    work = folder(tmp_path, problem=problem)
    bench = problem / "openmp" / "bench.c"
    bench.write_text(bench.read_text() + "/* Changed after the work folder was assembled. */\n")

    ordinary = run_check(work, problem, "openmp")
    assert ordinary.failure_stage == "tampered", ordinary.log
    assert "bench.c differs from the copy made from the problem's own files" in ordinary.log

    result = run_check(work, problem, "openmp", as_assembled=True)
    assert (result.passed, result.failure_stage) == (True, "none"), result.log
    assert "[check] Judged as assembled" in result.log
    assert "bench.c differs from the problem's current copy" in result.log
    assert "answers agree with the known-good routine's" in result.log
    assert "[check] This work folder was judged as assembled." in result.log


def test_an_old_folder_with_kernel_and_no_warm_up_is_judged_as_assembled(tmp_path):
    problem = problem_copy(tmp_path, old_style=True)
    work = folder(tmp_path, problem=problem)
    assert "KERNEL" in (work / "Makefile").read_text()
    # The problem then moves on to the present style.
    for name in ("Makefile", "bench.c"):
        shutil.copy2(FIXTURE / "openmp" / name, problem / "openmp" / name)

    assert run_check(work, problem, "openmp").failure_stage == "tampered"
    result = run_check(work, problem, "openmp", as_assembled=True)
    assert (result.passed, result.failure_stage) == (True, "none"), result.log
    assert "Makefile differs from the problem's current copy" in result.log
    assert "bench.c differs from the problem's current copy" in result.log


def test_judging_as_assembled_still_applies_the_present_rules(tmp_path):
    problem = problem_copy(tmp_path, old_style=True)
    work = folder(tmp_path, WRONG_AT_THE_BENCHMARK_SIZE, problem=problem)
    result = run_check(work, problem, "openmp", as_assembled=True)
    assert (result.passed, result.failure_stage) == (False, "benchmark"), result.log


def test_a_changed_trusted_file_is_still_tampered_when_judged_as_assembled(tmp_path):
    work = folder(tmp_path)
    (work / "bench.c").write_text((work / "bench.c").read_text() + "/* edited */\n")
    result = run_check(work, FIXTURE, "openmp", as_assembled=True)
    assert result.failure_stage == "tampered", result.log
    assert "bench.c has been changed" in result.log


def test_a_folder_without_a_fingerprint_file_is_refused(tmp_path):
    work = folder(tmp_path)
    (work / TRUSTED_MANIFEST).unlink()
    with pytest.raises(CheckError, match="has no fingerprint file"):
        run_check(work, FIXTURE, "openmp", as_assembled=True)


@pytest.mark.parametrize("name", ["GNUmakefile", "makefile"])
def test_a_folder_with_a_makefile_that_make_reads_first_is_refused(tmp_path, name):
    work = folder(tmp_path)
    (work / name).write_text("check:\n\ttrue\n")
    with pytest.raises(CheckError, match=f"holds {name}"):
        run_check(work, FIXTURE, "openmp", as_assembled=True)


def test_a_fingerprint_file_that_does_not_list_the_header_is_refused(tmp_path):
    work = folder(tmp_path)
    manifest = json.loads((work / TRUSTED_MANIFEST).read_text())
    del manifest["sha256"]["midpoint.h"]
    (work / TRUSTED_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(CheckError, match="does not list midpoint.h"):
        run_check(work, FIXTURE, "openmp", as_assembled=True)


def test_a_fingerprint_file_that_names_an_unexpected_file_is_refused(tmp_path):
    """An extra header listed in the fingerprint file would be copied in and could switch the
    recorder off, so judging as assembled refuses it."""
    work = folder(tmp_path)
    (work / "extra.h").write_text("/* not a trusted file */\n")
    manifest = json.loads((work / TRUSTED_MANIFEST).read_text())
    manifest["sha256"]["extra.h"] = "0" * 64
    (work / TRUSTED_MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(CheckError, match="extra.h, which are not trusted files"):
        run_check(work, FIXTURE, "openmp", as_assembled=True)


def test_the_callers_openmp_and_make_settings_do_not_reach_the_judgement(monkeypatch):
    """A thread count or a make option in the caller's shell must not change a verdict."""
    for name, value in (("OMP_NUM_THREADS", "1"), ("OMP_PROC_BIND", "true"),
                        ("MAKEFLAGS", "-k"), ("MAKEFILES", "evil.mk")):  # fmt: skip
        monkeypatch.setenv(name, value)
    env = check_module._environment("openmp")
    for name in ("OMP_NUM_THREADS", "OMP_PROC_BIND", "MAKEFLAGS", "MAKEFILES"):
        assert name not in env


def test_the_check_command_takes_the_option(tmp_path, capsys):
    problem = problem_copy(tmp_path)
    work = folder(tmp_path, problem=problem)
    bench = problem / "openmp" / "bench.c"
    bench.write_text(bench.read_text() + "/* changed */\n")
    args = ["check", str(work), "--problem", str(problem), "--backend", "openmp"]
    assert cli.main(args) == 1
    capsys.readouterr()
    assert cli.main([*args, "--as-assembled"]) == 0
    out = capsys.readouterr().out
    assert "This work folder was judged as assembled." in out
    assert out.splitlines()[-1].startswith("PASS: correct")
