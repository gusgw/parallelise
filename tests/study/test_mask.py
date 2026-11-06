"""Tests for masking what an agent sees of a routine: its Python, its serial C and its header."""

import re
from pathlib import Path

import pytest

from parallelise import generate, notebook
from parallelise.errors import ParalleliseError
from parallelise.study.mask import mask_sources, strip_c_comments

FIXTURE = Path(__file__).parent.parent / "fixtures" / "midpoint"


@pytest.fixture
def masked():
    nb = FIXTURE / "midpoint.ipynb"
    decl = notebook.declaration(nb)
    return mask_sources(notebook.python_source(nb), (FIXTURE / "serial.c").read_text(),
                        generate.header(decl, "problem_a"), "midpoint", "problem_a")  # fmt: skip


def test_the_routine_is_named_by_its_alias_everywhere(masked):
    for text in masked:
        assert "midpoint" not in text.lower() and "problem_a" in text


def test_comments_and_docstrings_go_but_the_header_keeps_its_rules(masked):
    python, serial_c, header = masked
    assert '"""' not in python and "#" not in python
    assert "/*" not in serial_c and "//" not in serial_c
    assert "nothing from one call to the next" in header


def test_library_names_stay(masked):
    python, serial_c, _ = masked
    assert "np." in python and "#include <math.h>" in serial_c


def test_every_defined_name_is_neutral_and_the_same_in_all_three_sources():
    python = "import numpy as np\ndef scan(x, steps=2):\n    total = np.sum(x, axis=0)\n" \
             "    return helper(total, steps=steps)\ndef helper(v, steps):\n    return v * steps\n"
    c_source = '#include "scan.h"\nvoid scan(const double *x, double *out, long n)\n' \
               "{ double total = 0.0; for (long i = 0; i < n; i++) total += x[i];\n" \
               "  out[0] = total; }\n"
    header = "void problem_c(const double *x, double *out, long n);\n"
    py, c, h = mask_sources(python, c_source, header, "scan", "problem_c")
    for name in ("total", "helper", "steps"):
        assert not re.search(rf"\b{name}\b", py + c + h), name
    x = re.search(r"problem_c\(const double \*(\w+)", h).group(1)
    assert re.match(r"v\d+$", x) and f"def problem_c({x}," in py and f"double *{x}" in c
    assert "axis=0" in py and '#include "problem_c.h"' in c


@pytest.mark.parametrize("alias", ["bad name", "1abc", "x", "double"])
def test_an_alias_that_is_not_a_plain_unused_name_is_refused(alias):
    with pytest.raises(ParalleliseError, match="alias"):
        mask_sources("def f(x):\n    return x\n", "", "", "f", alias)


def test_c_comments_go_and_literals_stay():
    source = 'int x = 1; /* gone */ const char *s = "a // kept /* kept */";\n// gone\n'
    text = strip_c_comments(source)
    assert '"a // kept /* kept */"' in text and "gone" not in text
