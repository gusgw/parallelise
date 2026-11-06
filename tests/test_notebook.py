"""Tests for running a notebook's functions: its declaration, its inputs and its answers."""

import json
from pathlib import Path

import pytest

from parallelise import notebook
from parallelise.routine import read_inputs, read_results

FIXTURE = Path(__file__).parent / "fixtures" / "midpoint" / "midpoint.ipynb"


def write_notebook(path: Path, source: str) -> Path:
    cell = {"cell_type": "code", "metadata": {}, "outputs": [], "source": source}
    path.write_text(json.dumps({"cells": [cell], "metadata": {}, "nbformat": 4,
                                "nbformat_minor": 5}))  # fmt: skip
    return path


def test_the_fixture_declares_its_routine():
    decl = notebook.declaration(FIXTURE)
    assert decl.name == "midpoint" and [a.name for a in decl.arguments] == ["a", "b", "n"]


def test_inputs_come_from_the_seed_and_answers_from_the_function(tmp_path):
    decl = notebook.declaration(FIXTURE)
    calls = [(1000, 5, tmp_path / "one"), (1000, 5, tmp_path / "two")]
    calls.append((1000, 6, tmp_path / "three"))
    notebook.write_input_files(FIXTURE, calls)
    one, two, three = (read_inputs(c[2], decl) for c in calls)
    assert one == two and one != three
    notebook.write_answer_files(FIXTURE, [(tmp_path / "one", tmp_path / "answer")])
    assert 0.5 < abs(read_results(tmp_path / "answer", decl, {})["value"]) < 3.0


def test_a_function_whose_result_does_not_match_its_declaration_is_refused(tmp_path):
    path = write_notebook(tmp_path / "bad.ipynb", """
import numpy as np
def f(x):
    return x[:-1]
ROUTINE = {"function": f, "make_inputs": lambda size, rng: {"x": rng.uniform(size=size)},
           "arguments": {"x": "float64[n]"}, "results": {"out": "float64[n]"},
           "test_sizes": [3], "benchmark_size": 10, "tolerance": 1e-9}
""")
    with pytest.raises(notebook.NotebookError, match="shape"):
        notebook.declaration(path)


def test_a_notebook_without_a_declaration_is_refused(tmp_path):
    with pytest.raises(notebook.NotebookError, match="ROUTINE"):
        notebook.declaration(write_notebook(tmp_path / "none.ipynb", "x = 1\n"))


def test_the_python_shown_to_agents_is_the_function_and_what_it_uses(tmp_path):
    path = write_notebook(tmp_path / "f.ipynb", """
import numpy as np
SCALE = 2.0
UNUSED = 3.0
def helper(x):
    return SCALE * x
def f(x):
    return float(helper(x).sum())
data = np.ones(3)
print(f(data))
def make_inputs(size, rng):
    return {"x": rng.uniform(size=size)}
ROUTINE = {"function": f, "make_inputs": make_inputs, "arguments": {"x": "float64[n]"},
           "results": {"s": "float64"}, "test_sizes": [3], "benchmark_size": 10,
           "tolerance": 1e-9}
""")
    shown = notebook.python_source(path)
    for kept in ("import numpy as np", "SCALE = 2.0", "def helper", "def f"):
        assert kept in shown
    for dropped in ("UNUSED", "data", "print", "make_inputs", "ROUTINE"):
        assert dropped not in shown
