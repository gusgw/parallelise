"""Tests for masking what an agent sees of a routine: its Python, its serial C and its header."""

import re
from pathlib import Path

import pytest

from parallelise import convert, generate, notebook
from parallelise.errors import ParalleliseError
from parallelise.study.mask import mask_sources, strip_c_comments

FIXTURE = Path(__file__).parent.parent / "fixtures" / "midpoint"


@pytest.fixture
def masked():
    nb = FIXTURE / "midpoint.ipynb"
    decl = notebook.declaration(nb)
    return mask_sources(notebook.python_source(nb), (FIXTURE / "serial.c").read_text(),
                        generate.header(decl, "problem_a"), "midpoint", "problem_a",
                        ["a", "b", "n", "value"])  # fmt: skip


def test_the_routine_is_named_by_its_alias_everywhere(masked):
    for text in masked[:3]:
        assert "midpoint" not in text.lower() and "problem_a" in text


def test_comments_and_docstrings_go_but_the_header_keeps_its_rules(masked):
    python, serial_c, header, _ = masked
    assert '"""' not in python and "#" not in python
    assert "/*" not in serial_c and "//" not in serial_c
    assert "nothing from one call to the next" in header


def test_library_names_stay(masked):
    python, serial_c, _, _ = masked
    assert "np." in python and "#include <math.h>" in serial_c


def test_every_defined_name_is_neutral_and_the_same_in_all_three_sources():
    python = "import numpy as np\ndef scan(x, steps=2):\n    total = np.sum(x, axis=0)\n" \
             "    return helper(total, steps=steps)\ndef helper(v, steps):\n    return v * steps\n"
    c_source = '#include "scan.h"\nvoid scan(const double *x, double *out, long n)\n' \
               "{ double total = 0.0; for (long i = 0; i < n; i++) total += x[i];\n" \
               "  out[0] = total; }\n"
    header = "void problem_c(const double *x, double *out, long n);\n"
    py, c, h, _ = mask_sources(python, c_source, header, "scan", "problem_c")
    for name in ("total", "helper", "steps"):
        assert not re.search(rf"\b{name}\b", py + c + h), name
    x = re.search(r"problem_c\(const double \*(\w+)", h).group(1)
    assert re.match(r"v\d+$", x) and f"def problem_c({x}," in py and f"double *{x}" in c
    assert "axis=0" in py and '#include "problem_c.h"' in c


def test_the_declaration_under_the_masked_names_gives_the_masked_header(masked):
    decl = notebook.declaration(FIXTURE / "midpoint.ipynb")
    shown = decl.renamed(masked[3])
    assert shown.name == "problem_a" and generate.header(shown, "problem_a") == masked[2]


@pytest.mark.slow
def test_the_check_of_a_masked_run_names_nothing_that_masking_hid(tmp_path):
    nb = FIXTURE / "midpoint.ipynb"
    folder = tmp_path / "work"
    folder.mkdir()
    shown = convert.prepare(nb, notebook.declaration(nb), FIXTURE / "serial.c", "openmp", folder,
                            alias="problem_a")  # fmt: skip
    tools = convert.toolset(nb, shown, folder, shown.name, "openmp", tmp_path / "cache")
    a, b, n = (v.name for v in shown.arguments)
    wrong = (f'#include "problem_a.h"\ndouble problem_a(double {a}, double {b}, long {n})\n'
             f"{{ (void){a}; (void){b}; (void){n}; return -1.0; }}\n")
    tools.run("write_file", {"path": "problem_a.c", "content": wrong})
    told = tools.run("check", {})
    assert f"{shown.results[0].name} differs" in told and "value" not in told
    tools.run("write_file", {"path": "problem_a.c", "content": wrong.replace("long", "int")})
    told = tools.run("check", {})
    prototype = re.search(r"double problem_a\(.*\);", (folder / "problem_a.h").read_text())
    assert "conflicting types" in told and prototype.group(0) in told and "double a," not in told


@pytest.mark.parametrize("alias", ["bad name", "1abc", "x", "double"])
def test_an_alias_that_is_not_a_plain_unused_name_is_refused(alias):
    with pytest.raises(ParalleliseError, match="alias"):
        mask_sources("def f(x):\n    return x\n", "", "", "f", alias)


def test_c_comments_go_and_literals_stay():
    source = 'int x = 1; /* gone */ const char *s = "a // kept /* kept */";\n// gone\n'
    text = strip_c_comments(source)
    assert '"a // kept /* kept */"' in text and "gone" not in text
