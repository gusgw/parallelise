"""Tests for a routine's declaration: its C prototype, its files and the comparison of results."""

import numpy as np
import pytest

from parallelise.routine import (
    DeclarationError,
    c_prototype,
    compare,
    parse_declaration,
    read_inputs,
    read_results,
    write_inputs,
    write_results,
)


def declare(arguments, results, **extra):
    routine = {"arguments": arguments, "results": results, "test_sizes": [1, 10],
               "benchmark_size": 100, "tolerance": 1e-9, **extra}  # fmt: skip
    return parse_declaration(routine, "f")


@pytest.mark.parametrize(
    ("arguments", "results", "extra", "prototype"),
    [
        ({"a": "float64", "n": "int64"}, {"v": "float64"}, {}, "double f(double a, long n)"),
        ({"x": "float64[n]"}, {"out": "float64[n]"}, {},
         "void f(const double *x, double *out, long n)"),
        ({"x": "float64[n]", "t": "float64"}, {"idx": "int64[<=n]"}, {},
         "void f(const double *x, double t, long *idx, long *idx_count, long n)"),
        ({"u": "float64[n, n]", "iters": "int32"}, {"u": "float64[n, n]"},
         {"c_order": ["u", "n", "iters"]}, "void f(double *u, long n, int iters)"),
        ({"x": "float64[n]", "nbins": "int32"}, {"counts": "int64[nbins]"},
         {"c_order": ["x", "n", "counts", "nbins"]},
         "void f(const double *x, long n, long *counts, int nbins)"),
        ({"p": "float64[n, 3]"}, {"a": "float64[n, 3]"}, {},
         "void f(const double *p, double *a, long n)"),
    ],
)  # fmt: skip
def test_the_prototype_follows_from_the_declaration(arguments, results, extra, prototype):
    assert c_prototype(declare(arguments, results, **extra)) == prototype


@pytest.mark.parametrize(
    ("arguments", "results", "extra"),
    [
        ({"double": "float64"}, {"v": "float64"}, {}),
        ({"drv_x": "float64"}, {"v": "float64"}, {}),
        ({"x": "complex128"}, {"v": "float64"}, {}),
        ({"x": "float64[n]"}, {"v": "float64[m]"}, {}),
        ({"x": "float64[n]"}, {"out": "float64[n]"}, {"c_order": ["x", "n"]}),
        ({"u": "float64[n]"}, {"u": "float32[n]"}, {}),
        ({"x": "float64[n, 0]"}, {"v": "float64"}, {}),
        ({"x": "float64[n, 2.5]"}, {"v": "float64"}, {}),
    ],
)  # fmt: skip
def test_a_declaration_that_cannot_work_is_refused(arguments, results, extra):
    with pytest.raises(DeclarationError):
        declare(arguments, results, **extra)


def test_a_renamed_declaration_keeps_everything_but_the_names():
    decl = declare({"u": "float64[n, 3]", "t": "float64"}, {"u": "float64[n, 3]",
                   "idx": "int64[<=n]"}, c_order=["u", "n", "t", "idx"])  # fmt: skip
    shown = decl.renamed({"f": "g", "u": "v1", "t": "v2", "n": "v3", "idx": "v4"})
    assert c_prototype(shown, "g") == ("void g(double *v1, long v3, double v2, long *v4, "
                                       "long *v4_count)")  # fmt: skip
    assert shown.updated == {"v1"} and set(shown.tolerance) == {"v1", "v4"}
    assert (shown.test_sizes, shown.benchmark_size) == (decl.test_sizes, decl.benchmark_size)


def test_inputs_and_results_survive_their_files(tmp_path):
    decl = declare({"x": "float64[n, m]", "k": "int32"}, {"idx": "int64[<=n]", "s": "float32"})
    x = np.arange(6.0).reshape(2, 3)
    assert write_inputs(tmp_path / "in", decl, {"x": x, "k": 7}) == {"n": 2, "m": 3}
    back = read_inputs(tmp_path / "in", decl)
    assert np.array_equal(back["x"], x) and back["k"] == 7
    write_results(tmp_path / "out", decl, {"idx": np.array([1]), "s": 2.5})
    results = read_results(tmp_path / "out", decl, {"n": 2, "m": 3})
    assert results["idx"].tolist() == [1] and results["s"] == 2.5


def test_a_function_named_like_the_c_library_is_refused():
    routine = {"arguments": {"x": "float64"}, "results": {"v": "float64"}, "test_sizes": [1],
               "benchmark_size": 1, "tolerance": 0}  # fmt: skip
    with pytest.raises(DeclarationError, match="Rename the function"):
        parse_declaration(routine, "round")


def test_a_dimension_may_be_a_whole_number(tmp_path):
    decl = declare({"p": "float64[n, 3]"}, {"a": "float64[n, 3]"})
    assert decl.sizes == ("n",)
    p = np.arange(6.0).reshape(2, 3)
    assert write_inputs(tmp_path / "in", decl, {"p": p}) == {"n": 2}
    assert np.array_equal(read_inputs(tmp_path / "in", decl)["p"], p)
    with pytest.raises(DeclarationError, match="declared shape"):
        write_inputs(tmp_path / "in", decl, {"p": np.zeros((2, 4))})


def test_a_result_file_of_the_wrong_length_is_refused(tmp_path):
    decl = declare({"x": "float64[n]"}, {"out": "float64[n]"})
    (tmp_path / "out").write_bytes(np.zeros(3).tobytes())
    with pytest.raises(DeclarationError):
        read_results(tmp_path / "out", decl, {"n": 2})


def test_results_are_compared_by_kind_shape_and_tolerance():
    decl = declare({"x": "float64[n]"}, {"out": "float64[n]", "c": "int64[n]"},
                   tolerance={"out": 1e-6, "c": 0})  # fmt: skip
    want = {"out": np.array([1.0, -100.0]), "c": np.array([1, 2])}
    assert compare(decl, {"out": np.array([1.0, -100.00001]), "c": np.array([1, 2])}, want) == []
    assert compare(decl, {"out": np.array([1.1, -100.0]), "c": np.array([1, 2])}, want)
    assert compare(decl, {"out": np.array([np.nan, -100.0]), "c": np.array([1, 2])}, want)
    assert compare(decl, {"out": np.array([1.0]), "c": np.array([1, 2])}, want)
    assert compare(decl, {"out": want["out"], "c": np.array([1, 3])}, want)


def test_an_integer_result_needs_no_tolerance():
    decl = declare({"x": "float64[n]"}, {"out": "float64[n]", "c": "int64[n]"},
                   tolerance={"out": 1e-6})  # fmt: skip
    assert decl.tolerance["c"] == ("relative", 0.0)
    routine = {"arguments": {"x": "float64[n]"}, "results": {"c": "int64[n]"},
               "test_sizes": [1], "benchmark_size": 10}  # fmt: skip
    assert parse_declaration(routine, "f").tolerance["c"] == ("relative", 0.0)
    with pytest.raises(DeclarationError, match="need none"):
        declare({"x": "float64[n]"}, {"out": "float64[n]"}, tolerance={})


def test_an_absolute_tolerance_does_not_scale():
    decl = declare({"x": "float64"}, {"e": "float64"}, tolerance=("absolute", 0.01))
    assert compare(decl, {"e": np.float64(5.005)}, {"e": np.float64(5.0)}) == []
    assert compare(decl, {"e": np.float64(5.02)}, {"e": np.float64(5.0)})
