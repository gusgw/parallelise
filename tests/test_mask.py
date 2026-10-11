"""Tests for masking, using the synthetic fixture so they do not depend on real problems."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from parallelise.assemble import assemble_work_dir
from parallelise.gate import run_gate
from parallelise.mask import (
    NEUTRAL_SENTENCE,
    WORD_LIST,
    mask_problem,
    strip_c_comments,
    strip_makefile_comments,
)

FIXTURE = Path(__file__).parent / "fixtures" / "midpoint"
ALIAS = "kernel_a"


def run_notebook_oracle(path: Path) -> dict:
    notebook = json.loads(path.read_text())
    namespace: dict = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)  # noqa: S102 - our own fixture
    return namespace["ORACLE"]


@pytest.fixture
def masked(tmp_path):
    dest = tmp_path / "masked"
    report = mask_problem(FIXTURE, dest, ALIAS)
    return dest, report


def test_file_names_use_the_alias(masked):
    dest, report = masked
    names = {str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()}
    assert names == {
        "kernel_a.ipynb", "reference.h", "openmp/kernel_a.h", "openmp/bench.c",
        "openmp/Makefile", "openmp/reference/kernel_a.c", "openmp/reference/test_kernel_a.c",
    }  # fmt: skip
    assert ("midpoint.ipynb", "kernel_a.ipynb") in report.renamed_files


def test_no_trace_of_the_original_name_or_comments(masked):
    dest, _ = masked
    for path in dest.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text()
        assert "midpoint" not in text.lower(), path
        if path.suffix == ".c" or path.suffix == ".h":
            assert "//" not in text and "/*" not in text, path
        if path.name == "Makefile":
            assert "#" not in text, path
    assert "KERNEL  := kernel_a" in (dest / "openmp/Makefile").read_text()
    assert "#ifndef KERNEL_A_H" in (dest / "openmp/kernel_a.h").read_text()


def test_word_list_identifiers_are_renamed(masked):
    dest, report = masked
    source = (dest / "openmp/reference/kernel_a.c").read_text()
    assert "integral_sum" not in source
    assert re.search(r"double mk\d+_sum = 0.0;", source)
    assert "#pragma omp parallel for reduction(+ : mk" in source
    assert any(s.original == "integral_sum" for s in report.substitutions)


def test_notebook_markdown_and_oracle(masked):
    dest, _ = masked
    notebook = json.loads((dest / "kernel_a.ipynb").read_text())
    markdown = [c for c in notebook["cells"] if c["cell_type"] == "markdown"]
    assert len(markdown) == 1
    assert "".join(markdown[0]["source"]) == NEUTRAL_SENTENCE
    original = run_notebook_oracle(FIXTURE / "midpoint.ipynb")
    after = run_notebook_oracle(dest / "kernel_a.ipynb")
    assert after.pop("kernel") == ALIAS
    assert original.pop("kernel") == "midpoint"
    assert after == original  # numbers, parameters and tolerance are unchanged


def test_library_names_survive(masked):
    dest, _ = masked
    code = "\n".join(
        "".join(c["source"])
        for c in json.loads((dest / "kernel_a.ipynb").read_text())["cells"]
        if c["cell_type"] == "code"
    )
    assert "from math import pi, sin" in code
    assert "math.pi" in code
    assert '"""' not in code and "#" not in code


def test_report_lists_substitutions(masked):
    _, report = masked
    assert report.kernel == "midpoint" and report.alias == ALIAS
    assert report.comments_removed > 0 and report.markdown_cells_replaced == 1
    text = report.summary()
    assert "midpoint -> kernel_a" in text
    assert "MIDPOINT_H -> KERNEL_A_H" in text


def test_bad_aliases_are_refused(tmp_path):
    with pytest.raises(ValueError):
        mask_problem(FIXTURE, tmp_path / "x", "bad name")
    with pytest.raises(ValueError):
        mask_problem(FIXTURE, tmp_path / "x", "jacobi")


def test_c_comment_stripping_respects_strings():
    source = 'int x = 1; /* gone */ const char *s = "a // not a comment /* nor this */";\n// gone\n'
    text, removed = strip_c_comments(source)
    assert removed == 2
    assert '"a // not a comment /* nor this */"' in text
    assert "gone" not in text


def test_makefile_comment_stripping():
    text, removed = strip_makefile_comments("# top\nX := 1  # trailing\nY := 'a # b'\n")
    assert removed == 2
    assert text == "X := 1\nY := 'a # b'\n"


def test_library_identifiers_in_c_are_protected(tmp_path):
    problem = tmp_path / "scan"
    (problem / "mpi").mkdir(parents=True)
    (problem / "scan.ipynb").write_text(
        json.dumps({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5})
    )
    (problem / "mpi" / "bench.c").write_text(
        "#pragma omp scan inclusive(prefix_total) // note\n"
        "double v = M_PI * jacobi_step(MPI_Scan, n_bodies);\n"
        "double w = thrust::inclusive_scan(REF_PI_VALUE);\n"
    )
    out = tmp_path / "out"
    mask_problem(problem, out, "kernel_b")
    text = (out / "mpi" / "bench.c").read_text()
    assert re.search(r"#pragma omp scan inclusive\(mk\d+_total\)", text)
    assert "M_PI" in text and "MPI_Scan" in text and "REF_PI_VALUE" in text
    assert "thrust::inclusive_scan" in text
    assert "jacobi" not in text and "bodies" not in text
    assert "_step(" in text


def test_word_list_covers_the_nine_problems():
    for word in ("jacobi", "heat", "stencil", "midpoint", "integrate", "pi", "matmul", "matrix",
                 "histogram", "scan", "prefix", "nbody", "body", "cg", "conjugate", "gradient",
                 "decay", "reservoir"):  # fmt: skip
        assert word in WORD_LIST


@pytest.mark.slow
@pytest.mark.skipif(
    shutil.which("gcc") is None or shutil.which("make") is None, reason="needs gcc and make"
)
def test_masked_reference_builds_and_passes(masked, tmp_path):
    dest, _ = masked
    probe = subprocess.run(
        ["gcc", "-fopenmp", "-x", "c", "-", "-o", "/dev/null"],
        input="#include <omp.h>\nint main(void){return 0;}\n",
        capture_output=True, text=True, timeout=60, check=False,
    )  # fmt: skip
    if probe.returncode != 0:
        pytest.skip("gcc has no OpenMP")
    work = assemble_work_dir(dest, "openmp", tmp_path / "w", include_reference=True)
    result = run_gate(work, "openmp", timeout_s=240)
    assert result.passed, result.log
