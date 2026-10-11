"""Tests for oracle extraction, validation and the reference files."""

import json
from pathlib import Path

import pytest

from parallelise.errors import OracleError
from parallelise.oracle import (
    c_literal,
    extract_oracle,
    reorder_spread,
    write_reference,
)


def make_notebook(path: Path, *cells: str) -> Path:
    nb = {
        "cells": [{"cell_type": "markdown", "source": ["# title"]}]
        + [{"cell_type": "code", "source": c.splitlines(keepends=True)} for c in cells],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


GOOD = (
    "N = 1000\n"
    "terms = [1.0 / (i + 1) for i in range(N)]\n",
    "ORACLE = {'kernel': 'harmonic', 'params': {'N': N},\n"
    "          'expected': {'TOTAL': sum(terms)}, 'terms': {'TOTAL': terms}, 'tol': 1e-6}\n",
)


def test_extract_and_write_reference(tmp_path):
    nb = make_notebook(tmp_path / "harmonic.ipynb", *GOOD)
    oracle = extract_oracle(nb)
    assert oracle.kernel == "harmonic" and oracle.source == "harmonic.ipynb"
    assert oracle.reorder_spread["TOTAL"] >= 0
    out = tmp_path / "out"
    json_path, header_path = write_reference(oracle, out)
    header = header_path.read_text().splitlines()
    assert header[0] == (
        "/* generated from harmonic.ipynb by make_oracle.py -- the oracle the C must match */"
    )
    assert header[1:3] == ["#ifndef REFERENCE_H", "#define REFERENCE_H"]
    assert "#define REF_N 1000L" in header
    assert any(ln.startswith("#define REF_TOTAL 7.4854") for ln in header)
    assert "#define REF_TOL 1e-06" in header
    assert header[-1] == "#endif"
    data = json.loads(json_path.read_text())
    assert list(data) == ["source", "kernel", "params", "expected", "tol", "reorder_spread"]
    assert data["params"] == {"N": 1000}


def test_c_literals():
    assert c_literal(5) == "5L"
    assert c_literal(5, force_double=True) == "5.0"
    assert c_literal(1e-6) == "1e-06"
    assert c_literal(0.1) == "0.1"
    with pytest.raises(OracleError):
        c_literal(True)


def test_reorder_spread_is_zero_for_exact_terms():
    assert reorder_spread([1.0, 2.0, 3.0]) == 0.0
    assert reorder_spread([1e16, 1.0, -1e16, 1.0] * 4) > 0


def oracle_src(params="{'N': 1}", expected="{'X': 1.0}", tol="1", kernel="'k'", extra=""):
    return (
        f"ORACLE = {{'kernel': {kernel}, 'params': {params}, "
        f"'expected': {expected}, 'tol': {tol}{extra}}}"
    )


@pytest.mark.parametrize(
    "source, words",
    [
        ("ORACLE = 3", "dictionary"),
        ("ORACLE = {'kernel': 'k', 'params': {'N': 1}, 'expected': {'X': 1.0}}", "'tol'"),
        (oracle_src(kernel="''"), "kernel"),
        (oracle_src(expected="{'N': 1.0}"), "repeated"),
        (oracle_src(params="{'TOL': 1}"), "repeated"),
        (oracle_src(params="{'1N': 1}"), "macro"),
        (oracle_src(expected="{'X': float('nan')}"), "finite"),
        (oracle_src(expected="{'X': True}"), "finite number"),
        (oracle_src(tol="0"), "positive"),
        (oracle_src(extra=", 'terms': {'Y': [1.0]}"), "no matching"),
        (oracle_src(expected="{'X': 5.0}", tol="0.1", extra=", 'terms': {'X': [1.0, 2.0]}"),
         "do not produce"),
        (oracle_src(expected="{'X': 8.0}", tol="1e-30",
                    extra=", 'terms': {'X': [1e16, 1.0, -1e16, 1.0] * 4}"), "floor"),
    ],
)
def test_validation_rules(tmp_path, source, words):
    nb = make_notebook(tmp_path / "bad.ipynb", source)
    with pytest.raises(OracleError, match=words):
        extract_oracle(nb)


def test_missing_oracle_and_failing_cell(tmp_path):
    with pytest.raises(OracleError, match="no ORACLE"):
        extract_oracle(make_notebook(tmp_path / "a.ipynb", "x = 1"))
    with pytest.raises(OracleError, match="cell 1 failed: ZeroDivisionError"):
        extract_oracle(make_notebook(tmp_path / "b.ipynb", "1/0"))
    with pytest.raises(OracleError, match="does not exist"):
        extract_oracle(tmp_path / "nothing.ipynb")


def test_notebook_runs_in_a_child_with_timeout(tmp_path):
    nb = make_notebook(tmp_path / "slow.ipynb", "import time\ntime.sleep(30)")
    with pytest.raises(OracleError, match="longer than"):
        extract_oracle(nb, timeout_s=1)
    # a notebook that exits does not end the calling process
    nb = make_notebook(tmp_path / "quit.ipynb", "import sys\nsys.exit(0)")
    with pytest.raises(OracleError):
        extract_oracle(nb)


def test_shipped_problems_regenerate_unchanged():
    root = Path(__file__).resolve().parent.parent / "problems"
    notebook = root / "integrate" / "integrate.ipynb"
    if not notebook.exists():
        pytest.skip("problems folder not present")
    from parallelise.oracle import render_header, render_json

    oracle = extract_oracle(notebook)
    assert render_header(oracle) == (root / "integrate" / "reference.h").read_text()
    assert render_json(oracle) == (root / "integrate" / "reference.json").read_text()
