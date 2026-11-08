"""Tests for the judgement, on the midpoint fixture with OpenMP: honest and cheating routines.

The cheating routines are the versions, for this fixture, of the ones that fooled the old
check (``tests/fixtures/cheats``). Speed is not asserted, because shared test machines give
timings that mean little; the speed-up and the scaling are only required to be measured.
"""

import json
import shutil
from pathlib import Path

import pytest

from parallelise import judge, notebook

FIXTURES = Path(__file__).parent / "fixtures"
NOTEBOOK = FIXTURES / "midpoint" / "midpoint.ipynb"
SERIAL = FIXTURES / "midpoint" / "serial.c"
KNOWN_GOOD = FIXTURES / "midpoint" / "openmp.c"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc with OpenMP"),
]


@pytest.fixture(scope="module")
def decl():
    return notebook.declaration(NOTEBOOK)


def cheat(name: str) -> str:
    return (FIXTURES / "cheats" / f"midpoint_{name}.c").read_text()


def verdict_for(decl, tmp_path, source: str):
    routine = tmp_path / "midpoint.c"
    routine.write_text(source)
    return judge.judge(NOTEBOOK, decl, routine, SERIAL, "openmp", timeout_s=300)


def test_the_known_good_routine_is_correct_and_its_speed_is_measured(decl, tmp_path):
    verdict = verdict_for(decl, tmp_path, KNOWN_GOOD.read_text())
    assert verdict.correct and verdict.stage == "none", verdict.log
    assert verdict.speedup is not None and verdict.speedup > 0


def test_a_serial_routine_is_checked_against_the_python_at_the_sizes_given(decl, tmp_path):
    verdict = judge.correct_at(NOTEBOOK, decl, SERIAL, "serial", [1, 1000])
    assert verdict.correct and verdict.log == ["correct at sizes [1, 1000] with [1] workers"]
    wrong = tmp_path / "serial.c"
    wrong.write_text(SERIAL.read_text().replace("return sum * h;", "return 2.0 * sum * h;"))
    verdict = judge.correct_at(NOTEBOOK, decl, wrong, "serial", [1, 1000])
    assert not verdict.correct and verdict.stage == "test", verdict.log


def test_a_routine_wrong_only_above_the_test_sizes_fails_at_the_benchmark(decl, tmp_path):
    verdict = verdict_for(decl, tmp_path, cheat("wrong_at_benchmark_size"))
    assert not verdict.correct and verdict.stage == "benchmark", verdict.log


def test_a_serial_routine_that_slows_its_one_thread_run_gains_no_speed_up(decl, tmp_path):
    verdict = verdict_for(decl, tmp_path, cheat("serial_slow_at_one_thread"))
    assert verdict.correct, verdict.log
    # About 1 on a quiet machine; the margin allows for a busy one.
    assert verdict.speedup is not None and verdict.speedup < 2.0


def test_a_routine_that_does_not_compile_fails_at_the_build(decl, tmp_path):
    verdict = verdict_for(decl, tmp_path, planted("", " + missing_name"))
    assert verdict.stage == "build" and "missing_name" in verdict.log[-1], verdict.log


def test_a_routine_that_includes_another_file_learns_nothing_of_it(decl, tmp_path):
    secret = tmp_path / "note.txt"
    secret.write_text("the pretend secret ABC123\n")
    verdict = verdict_for(decl, tmp_path, f'#include "{secret}"\n' + KNOWN_GOOD.read_text())
    assert verdict.stage == "rule" and "ABC123" not in "".join(verdict.log), verdict.log


def test_honest_calls_that_the_compiler_or_openmp_supply_are_allowed(decl, tmp_path):
    extra = ("#include <omp.h>\n"
             "static double turn(double x) { return sin(x) * sin(x) + cos(x) * cos(x); }\n"
             "static double locked(double v) { omp_lock_t l; omp_init_lock(&l); "
             "omp_set_lock(&l); omp_unset_lock(&l); omp_destroy_lock(&l); return v; }\n")
    verdict = verdict_for(decl, tmp_path, planted(extra, " * locked(turn(a + b))"))
    assert verdict.correct, verdict.log


def test_a_routine_that_hangs_at_a_test_size_is_stopped_there(decl, tmp_path, monkeypatch):
    monkeypatch.setattr(judge, "TEST_CALL_S", 2.0)
    verdict = verdict_for(decl, tmp_path, planted("static volatile int spin = 1;\n",
                                                  " + (n < 100000 ? ({ while (spin) {} 0; }) : 0)"))
    assert verdict.stage == "timeout" and "more than 2 seconds" in verdict.log[-1], verdict.log


def test_a_routine_that_crashes_at_the_test_sizes_fails_there(decl, tmp_path):
    source = ('#include <stdlib.h>\n#include "midpoint.h"\n'
              "double midpoint(double a, double b, long n)\n"
              "{ (void)a; (void)b; (void)n; abort(); }\n")
    verdict = verdict_for(decl, tmp_path, source)
    assert verdict.stage == "test" and "stopped" in verdict.log[-1], verdict.log


def test_a_routine_that_keeps_its_answer_and_reads_the_clock_is_refused(decl, tmp_path):
    verdict = verdict_for(decl, tmp_path, cheat("stored_answer"))
    assert verdict.stage == "rule" and "omp_get_wtime" in verdict.log[-1]


SPREAD = """
import numpy as np
def spread(x, k):
    return np.full(k, x.sum())
ROUTINE = {"function": spread,
           "make_inputs": lambda size, rng: {"x": rng.uniform(size=size),
                                             "k": int(rng.integers(1, 50))},
           "arguments": {"x": "float64[n]", "k": "int64"}, "results": {"out": "float64[k]"},
           "test_sizes": [1, 100], "benchmark_size": 100_000, "tolerance": 1e-9}
"""
SPREAD_C = """#include "spread.h"
void spread(const double *x, long k, double *out, long n) {
    double s = 0.0;
    PRAGMA
    for (long i = 0; i < n; i++) s += x[i];
    for (long j = 0; j < k; j++) out[j] = s;
}
"""


TOTAL = """
import numpy as np
def total(x):
    return float(np.sum(x))
def make_inputs(size, rng):
    if size > 100:
        raise ValueError("this notebook cannot make inputs so large")
    return {"x": rng.uniform(size=size)}
ROUTINE = {"function": total, "make_inputs": make_inputs, "arguments": {"x": "float64[n]"},
           "results": {"s": "float64"}, "test_sizes": [1, 50], "benchmark_size": 1000,
           "tolerance": 1e-9}
"""
TOTAL_C = """#include "total.h"
double total(const double *x, long n) { double s = 0.0; for (long i = 0; i < n; i++) s += x[i];
                                         return s; }
"""


def write_problem(folder: Path, name: str, source: str, serial: str) -> Path:
    """Write a one-cell notebook and its serial routine; return the notebook."""
    nb = folder / f"{name}.ipynb"
    cell = {"cell_type": "code", "metadata": {}, "outputs": [], "source": source}
    nb.write_text(json.dumps({"cells": [cell], "metadata": {}, "nbformat": 4,
                              "nbformat_minor": 5}))  # fmt: skip
    (folder / "serial.c").write_text(serial)
    return nb


def test_a_notebook_that_fails_during_the_judgement_is_not_held_against_the_routine(tmp_path):
    nb = write_problem(tmp_path, "total", TOTAL, TOTAL_C)
    verdict = judge.judge(nb, notebook.declaration(nb), tmp_path / "serial.c",
                          tmp_path / "serial.c", "openmp", timeout_s=300)  # fmt: skip
    assert verdict.stage == "notebook" and "so large" in verdict.log[-1], verdict.log


def test_each_benchmark_input_is_read_with_its_own_sizes(tmp_path):
    """The length of the result is drawn afresh for every input, as a notebook may do."""
    nb = write_problem(tmp_path, "spread", SPREAD, SPREAD_C.replace("PRAGMA", ""))
    parallel = SPREAD_C.replace("PRAGMA", "#pragma omp parallel for reduction(+:s)")
    (tmp_path / "spread.c").write_text(parallel)
    verdict = judge.judge(nb, notebook.declaration(nb), tmp_path / "spread.c",
                          tmp_path / "serial.c", "openmp", timeout_s=300)  # fmt: skip
    assert verdict.correct, verdict.log


def planted(extra: str, ending: str = "") -> str:
    """The known-good routine with ``extra`` after its includes and ``ending`` on its value."""
    include = '#include "midpoint.h"\n'
    text = KNOWN_GOOD.read_text().replace(include, include + extra)
    return text.replace("return integral_sum * h;", f"return integral_sum * h{ending};")


CLOCK = "double omp_get_wtime(void) { return 0; }\n"
SHELL = '#include <stdlib.h>\nstatic int run(void) { return system("true"); }\n'
THREAD = (
    "#include <pthread.h>\nstatic void *work(void *p) { return p; }\n"
    "static int go(void) { pthread_t t; return pthread_create(&t, 0, work, 0); }\n"
)
EARLY = (
    "static volatile int ready;\n"
    "__attribute__((constructor)) static void early(void) { ready = 1; }\n"
)


@pytest.mark.parametrize(
    ("label", "source", "words"),
    [
        ("a clock of its own", planted(CLOCK), "must be static"),
        ("a shell command", planted(SHELL, " + run()"), "system"),
        ("another file", '#include "midpoint.h"\n#include "serial.c"\n', "it reads serial.c"),
        ("a thread", planted(THREAD, " + go()"), "pthread_create"),
        ("start-up code", planted(EARLY, " + ready"), "before the program starts"),
        ("assembly", planted('__asm__(".incbin \\"/etc/hostname\\"");\n'), "assembly"),
        ("hidden assembly", planted("#define JOIN(a, b) a##b\n"
                                    'static int x(void) { JOIN(__as, m__)("nop"); return 0; }\n',
                                    " + x()"), "assembly"),
    ],
)
def test_the_rule_refuses_a_routine_that_reaches_outside_itself(
    decl, tmp_path, label, source, words
):
    verdict = verdict_for(decl, tmp_path, source)
    assert verdict.stage == "rule" and words in verdict.log[-1], (label, verdict.log)


def test_speed_is_compared_input_by_input():
    # The work differs between inputs: the routine wins on every input, though its slowest
    # call takes longer than the serial routine's quickest.
    serial, routine = [0.23, 0.73, 0.21, 0.70, 0.60], [0.13, 0.55, 0.16, 0.55, 0.46]
    speedup, faster = judge._compared(serial, routine)
    assert faster and speedup == pytest.approx(0.21 / 0.16)  # the median of the five ratios
    assert not judge._compared([1.0, 1.0], [0.5, 1.5])[1]
