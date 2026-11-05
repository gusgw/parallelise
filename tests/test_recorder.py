"""Tests for the call recorder: reading the routine's signature, writing the C file, and
reading back and judging what it recorded."""

import shutil
import subprocess
from pathlib import Path

import pytest

from parallelise.recorder import (
    BUSY_ALLOWANCE,
    Call,
    PrototypeError,
    credit_speedups,
    parse_prototype,
    read_calls,
    recorder_source,
    verification_problems,
)

PROBLEMS = Path(__file__).resolve().parent.parent / "problems"
OPENMP_HEADERS = sorted(
    header
    for header in PROBLEMS.glob("*/openmp/*.h")
    if header.name not in ("reference.h", "problem.h")
)


def test_every_shipped_openmp_header_is_read():
    assert len(OPENMP_HEADERS) >= 9
    for header in OPENMP_HEADERS:
        prototype = parse_prototype(header.read_text())
        assert prototype.parameters, header


def test_the_routine_name_comes_from_the_header_not_the_file_name():
    prototype = parse_prototype((PROBLEMS / "cg/openmp/cg.h").read_text())
    assert prototype.name == "cgsolve"
    assert prototype.returns == "void"
    assert [(p.name, p.writable, p.element) for p in prototype.parameters] == [
        ("x", True, "double"), ("b", False, "double"), ("m", False, "long"),
        ("iters", False, "int"),
    ]  # fmt: skip


def test_a_returned_value_and_plain_parameters_are_read():
    prototype = parse_prototype("/* comment */\ndouble integrate(double a, double b, long n);\n")
    assert (prototype.name, prototype.returns) == ("integrate", "double")
    assert not any(p.pointer for p in prototype.parameters)


@pytest.mark.parametrize(
    "text",
    [
        "int a(int x);\nint b(int y);\n",  # two functions
        "#define X 1\n",  # none
        "void f(double x[], long n);\n",  # an array parameter
    ],
)
def test_a_header_the_recorder_cannot_handle_is_refused(text):
    with pytest.raises(PrototypeError):
        parse_prototype(text)


@pytest.mark.skipif(not shutil.which("gcc"), reason="needs gcc")
@pytest.mark.parametrize("header", OPENMP_HEADERS, ids=lambda h: h.parent.parent.name)
def test_the_recorder_compiles_with_warnings_as_errors(tmp_path, header):
    shutil.copy(header, tmp_path / header.name)
    source = tmp_path / "recorder.c"
    source.write_text(recorder_source(parse_prototype(header.read_text()), header.name))
    done = subprocess.run(
        ["gcc", "-O2", "-fopenmp", "-Wall", "-Wextra", "-Werror", "-std=c11", "-c",
         str(source), "-o", str(tmp_path / "recorder.o")],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr


def call(index, width, wall=1.0, cpu=1.0, checked=True, agrees=True, difference=0.0, scale=1.0):
    return Call(index, width, wall, cpu, checked, agrees, difference, scale, 0.5)


def test_recorded_calls_are_read_back(tmp_path):
    path = tmp_path / "calls.csv"
    path.write_text(
        "call,width,wall_s,cpu_s,checked,agrees,difference,scale,known_wall_s\n"
        "0,8,0.100000000,0.790000000,1,1,0,3.14,0.1\n"
        "1,1,0.800000000,0.800000000,1,0,2.5,3.14,0.8\n"
    )
    calls = read_calls(path)
    assert [c.width for c in calls] == [8, 1]
    assert calls[0].busy == pytest.approx(7.9)
    assert not calls[1].agrees


def test_an_overflowing_record_is_refused(tmp_path):
    path = tmp_path / "calls.csv"
    path.write_text("call,width,wall_s,cpu_s,checked,agrees,difference,scale,known_wall_s\n"
                    "4096,overflow\n")  # fmt: skip
    with pytest.raises(ValueError, match="more calls"):
        read_calls(path)


def test_answers_that_differ_are_reported():
    problems = verification_problems([call(0, 1), call(1, 4, agrees=False, difference=2.7,
                                                       scale=3.1)])  # fmt: skip
    assert problems == [
        "call 2 (4 threads) differs from the known-good routine by 2.7 where the answer is of "
        "size 3.1"
    ]
    assert verification_problems([]) == [
        "the benchmark made no call to the routine that could be compared"
    ]
    assert "not enough memory" in verification_problems([call(0, 1, checked=False)])[0]


def test_a_parallel_row_keeps_its_speedup():
    rows = [(1, 1.0, 1.0), (2, 0.5, 2.0), (4, 0.26, 3.85)]
    calls = [call(0, 4, 0.3, 1.2), call(1, 1, 1.0, 1.0), call(2, 2, 0.5, 1.0),
             call(3, 4, 0.26, 1.04)]  # fmt: skip
    credit = credit_speedups(rows, calls)
    assert (credit.best, credit.capped, credit.problems) == (3.85, [], [])


def test_a_serial_row_is_credited_with_its_busy_processors():
    rows = [(1, 4.0, 1.0), (4, 1.0, 4.0)]
    calls = [call(0, 1, 4.0, 4.0), call(1, 4, 1.0, 1.0)]
    credit = credit_speedups(rows, calls)
    assert credit.best == pytest.approx(1.0)
    assert "kept only 1.00 processors busy" in credit.capped[0]


def test_the_allowance_stays_below_the_lowest_floor():
    assert 1.0 < BUSY_ALLOWANCE < 1.25


def test_a_row_without_a_matching_call_is_a_problem():
    credit = credit_speedups([(1, 1.0, 1.0), (2, 0.5, 2.0)], [call(0, 1)])
    assert credit.problems == [
        "the table has a row for 2 threads, but the benchmark made no call to the routine "
        "with 2 threads"
    ]
    credit = credit_speedups([(1, 1.0, 1.0)], [call(0, 1, wall=3.0, cpu=3.0)])
    assert "but the call took 3.000000 s" in credit.problems[0]
