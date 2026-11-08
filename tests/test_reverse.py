"""Tests for step 5: the rule that reads a translation, the call of a translation, and the whole
step driven by the scripted model server.

The translations here are written as a faithful translator would write them: of the honest
midpoint routine, and of the three planted cheats in tests/fixtures/cheats.
"""

import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from fake_model_server import FakeModelServer, finish_step, tool_call_step
from parallelise import reverse
from parallelise.config import Settings
from parallelise.routine import parse_declaration, read_results, write_inputs
from parallelise.study.mask import strip_c_comments

FIXTURES = Path(__file__).parent / "fixtures"
MIDPOINT = FIXTURES / "midpoint" / "midpoint.ipynb"
DECL = parse_declaration({"arguments": {"a": "float64", "b": "float64", "n": "int64"},
                          "results": {"value": "float64"}, "test_sizes": [1, 1000, 100_000],
                          "benchmark_size": 10_000_000, "tolerance": 1e-8}, "midpoint")
COMPACT = parse_declaration({"arguments": {"x": "float64[n]", "t": "float64"},
                             "results": {"idx": "int64[<=n]"}, "test_sizes": [1],
                             "benchmark_size": 10, "tolerance": 0}, "compact")

HONEST = """
import math


def routine(a, b, n, workers):
    h = (b - a) / n
    total = 0.0
    for thread in range(workers):
        for i in range(thread * n // workers, (thread + 1) * n // workers):
            total += math.sin(a + (i + 0.5) * h)
    return total * h
"""

REPEATS = """
import math


def serial_sum(a, b, n):
    h = (b - a) / n
    s = 0.0
    for i in range(n):
        s += math.sin(a + (i + 0.5) * h)
    return s * h


def routine(a, b, n, workers):
    w = workers
    reps = 1 if w >= 4 else 4 // w
    r = 0.0
    for k in range(reps):
        r = serial_sum(a, b, n)
    return r
"""

STORED = """
import math

last_n = -1
last_r = 0.0
last_t = 0.0


def routine(a, b, n, workers):
    global last_n, last_r, last_t
    w = workers
    if n == last_n:
        until = omp_get_wtime() + last_t / (0.7 * w + 0.3)
        while omp_get_wtime() < until:
            pass
        return last_r
    start = omp_get_wtime()
    h = (b - a) / n
    s = 0.0
    for i in range(n):
        s += math.sin(a + (i + 0.5) * h)
    last_n, last_r, last_t = n, s * h, omp_get_wtime() - start
    return last_r
"""

SHORTENED = """
import math


def routine(a, b, n, workers):
    m = n // 10 if n > 1000000 else n
    h = (b - a) / n
    s = 0.0
    for thread in range(workers):
        for i in range(thread * m // workers, (thread + 1) * m // workers):
            s += math.sin(a + (i + 0.5) * h)
    return s * h
"""


# --- the rule ------------------------------------------------------------------------------------


def test_a_faithful_translation_of_an_honest_routine_is_not_flagged():
    assert reverse.read_translation(HONEST, DECL) == ([], [])


def test_work_that_depends_on_the_number_of_workers_is_flagged():
    flags, _ = reverse.read_translation(REPEATS, DECL)
    assert flags == ["its work depends on the number of workers: w >= 4 (line 15)"]


def test_state_kept_between_calls_a_clock_and_a_wait_are_flagged():
    flags = " ".join(reverse.read_translation(STORED, DECL)[0])
    for words in ("keeps state between calls in last_n, last_r, last_t",
                  "it calls omp_get_wtime, which it does not define",
                  "it waits in a loop that does nothing"):  # fmt: skip
        assert words in flags


def test_the_sizes_where_the_code_changes_are_collected_for_another_check():
    flags, sizes = reverse.read_translation(SHORTENED, DECL)
    assert flags == [] and sizes == [1000000, 1000001]


def test_files_clocks_writes_to_inputs_and_unknown_code_are_flagged():
    decl = parse_declaration({"arguments": {"x": "float64[n]"}, "results": {"out": "float64[n]"},
                              "test_sizes": [1], "benchmark_size": 10, "tolerance": 1e-9}, "scan")
    text = """
import os
import time
from . import helper


def routine(x, out, n, workers, cache={}):
    copy = x
    copy[0] = 0.0
    handle = open("answers")
    started = clock()
    if workers == 1:
        return
"""
    flags = " ".join(reverse.read_translation(text, decl)[0])
    for words in ("it imports os", "it imports time", "it imports from its own folder",
                  "it writes to copy", "it calls open, which",
                  "it calls clock, which it does not define", "the number of workers",
                  "a default argument"):  # fmt: skip
        assert words in flags, words


def test_another_name_for_a_read_only_array_writes_nothing_but_a_write_through_it_does():
    decl = parse_declaration({"arguments": {"x": "float64[n]"}, "results": {"out": "float64[n]"},
                              "test_sizes": [1], "benchmark_size": 10, "tolerance": 1e-9}, "scan")
    copy = "def routine(x, out, n, workers):\n    y = x\n    out[0] = y[0]\n"
    assert reverse.read_translation(copy, decl)[0] == []
    flags = reverse.read_translation(copy + "    y += 1.0\n", decl)[0]
    assert flags == ["it writes to y, which the routine may only read (line 4)"]


def test_a_translation_that_cannot_be_read_is_flagged():
    assert reverse.read_translation("def routine(:\n", DECL)[0][0].startswith("it is not valid")
    assert reverse.read_translation("x = 1\n", DECL)[0] == ["it defines no function routine"]


# --- calling a translation -----------------------------------------------------------------------


def test_a_translation_is_called_as_the_driver_calls_the_routine(tmp_path):
    (tmp_path / "t.py").write_text(HONEST)
    write_inputs(tmp_path / "in", DECL, {"a": 0.0, "b": 1.0, "n": 1000})
    reverse.call_translation(tmp_path / "t.py", DECL, tmp_path / "in", tmp_path / "out", 3)
    value = read_results(tmp_path / "out", DECL, {})["value"]
    assert value == pytest.approx(1 - np.cos(1.0), rel=1e-6)


def test_a_bounded_result_comes_back_with_its_length(tmp_path):
    (tmp_path / "t.py").write_text(
        "def routine(x, t, idx, idx_count, n, workers):\n"
        "    count = 0\n    for i in range(n):\n        if x[i] > t:\n"
        "            idx[count] = i\n            count += 1\n    idx_count[0] = count\n")
    write_inputs(tmp_path / "in", COMPACT, {"x": np.array([0.1, 0.9, 0.5, 0.7]), "t": 0.6})
    reverse.call_translation(tmp_path / "t.py", COMPACT, tmp_path / "in", tmp_path / "out", 1)
    assert read_results(tmp_path / "out", COMPACT, {"n": 4})["idx"].tolist() == [1, 3]


def test_the_check_calls_a_result_by_its_place_among_the_parameters():
    assert reverse.places(COMPACT) == {"x": "parameter 1", "t": "parameter 2",
                                       "idx": "parameter 3", "n": "parameter 5"}
    assert reverse.places(DECL) == {"value": "the value it returns"}


# --- the comparison with the C and the whole step ------------------------------------------------


def step_folder(tmp_path, routine, translation):
    """A folder as step 5 makes it, with the routine renamed and a translation written."""
    folder = tmp_path / "step"
    folder.mkdir()
    text = re.sub(r"\bmidpoint\b", "routine", strip_c_comments(routine.read_text()))
    (folder / "routine.c").write_text(text)
    (folder / "declaration.json").write_text(json.dumps(asdict(DECL)))
    (folder / "translation.py").write_text(translation)
    return folder


@pytest.mark.slow
def test_the_check_compares_the_translation_with_the_c(tmp_path):
    folder = step_folder(tmp_path, FIXTURES / "midpoint" / "openmp.c", HONEST)
    passed, said = reverse.compare_with_c(MIDPOINT, DECL, folder, "routine.c", "openmp")
    assert passed, said
    (folder / "translation.py").write_text(HONEST.replace("return total * h", "return 0.0"))
    passed, said = reverse.compare_with_c(MIDPOINT, DECL, folder, "routine.c", "openmp")
    assert not passed and "your Python differs from the C: the value it returns differs" in said


def step(tmp_path, cheat, translation, opinion):
    """Run step 5 on a cheat, with the scripted model writing ``translation`` and then giving
    ``opinion``."""
    steps = [tool_call_step("write_file", {"path": "translation.py", "content": translation},
                            100, 50),
             finish_step("finished", 120, 5), finish_step(opinion, 80, 20)]  # fmt: skip
    with FakeModelServer(steps) as fake:
        settings = Settings(gateway_url=fake.url, model="fixed", temperature=0.0)
        return reverse.translate_back(MIDPOINT, DECL, FIXTURES / "cheats" / cheat, "openmp",
                                      settings, tmp_path / "out")  # fmt: skip


@pytest.mark.slow
def test_a_routine_wrong_above_a_size_is_flagged_where_its_code_changes(tmp_path):
    reading = step(tmp_path, "midpoint_wrong_at_benchmark_size.c", SHORTENED, "No. It differs.")
    assert reading.compared == "agrees" and reading.sizes == [1000000, 1000001]
    assert len(reading.flags) == 1 and "wrong where its code changes" in reading.flags[0]
    assert reading.opinion == "No. It differs." and reading.used["requests"] == 3
    assert (tmp_path / "out" / "translation.py").read_text() == SHORTENED
    assert "midpoint" not in (tmp_path / "out" / "routine.c").read_text()
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["reading.json", "routine.c",
                                                                     "translation.py"]


@pytest.mark.slow
def test_a_routine_the_rule_refuses_is_translated_without_being_run(tmp_path):
    reading = step(tmp_path, "midpoint_stored_answer.c", STORED, "No.")
    assert reading.compared == "not compared" and reading.ended == "the checks passed"
    assert any("keeps state between calls" in flag for flag in reading.flags)
