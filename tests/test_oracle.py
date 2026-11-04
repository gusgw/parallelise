"""Tests for oracle extraction, validation and the reference files."""

import json
import logging
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
    "ORACLE = {'params': {'N': N},\n"
    "          'expected': {'TOTAL': sum(terms)}, 'terms': {'TOTAL': terms}, 'tol': 1e-6}\n",
)


def test_extract_and_write_reference(tmp_path):
    nb = make_notebook(tmp_path / "harmonic.ipynb", *GOOD)
    oracle = extract_oracle(nb)
    assert oracle.source == "harmonic.ipynb"
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
    assert list(data) == ["source", "params", "expected", "tol", "reorder_spread"]
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


def oracle_src(params="{'N': 1}", expected="{'X': 1.0}", tol="1", extra=""):
    return (
        f"ORACLE = {{'params': {params}, "
        f"'expected': {expected}, 'tol': {tol}{extra}}}"
    )


@pytest.mark.parametrize(
    "source, words",
    [
        ("ORACLE = 3", "dictionary"),
        ("ORACLE = {'params': {'N': 1}, 'expected': {'X': 1.0}}", "'tol'"),
        ("ORACLE = {'expected': {'X': 1.0}, 'tol': 1}", "'params'"),
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


def test_a_notebook_without_the_kernel_key_is_accepted(tmp_path, caplog):
    nb = make_notebook(tmp_path / "harmonic.ipynb", *GOOD)
    with caplog.at_level(logging.INFO, logger="parallelise.oracle"):
        oracle = extract_oracle(nb)
    assert oracle.params == {"N": 1000}
    assert not hasattr(oracle, "kernel")
    assert "no longer used" not in caplog.text


@pytest.mark.parametrize("value", ["'harmonic'", "''", "3"])
def test_an_old_notebook_with_the_kernel_key_is_still_accepted(tmp_path, caplog, value):
    old_oracle = GOOD[1].replace("ORACLE = {", f"ORACLE = {{'kernel': {value}, ")
    nb = make_notebook(tmp_path / "harmonic.ipynb", GOOD[0], old_oracle)
    with caplog.at_level(logging.INFO, logger="parallelise.oracle"):
        oracle = extract_oracle(nb)
    assert oracle.params == {"N": 1000}
    assert oracle.source == "harmonic.ipynb"
    notices = [r for r in caplog.records if "no longer used" in r.getMessage()]
    assert len(notices) == 1 and notices[0].levelno == logging.INFO
    json_path, _ = write_reference(oracle, tmp_path / "out")
    assert "kernel" not in json.loads(json_path.read_text())


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
