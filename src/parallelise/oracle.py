"""Run a research notebook and extract its oracle: the data the C code must reproduce.

The notebook's last code cell defines a plain dictionary named ORACLE:

    ORACLE = {"kernel": "matmul",                  # name of the C function and files
              "params": {"N": 256},                # inputs the C test must use
              "expected": {"CHECKSUM": checksum},  # the notebook's own computed output
              "tol": 0.5}                          # how closely the C must agree

It may also carry, for any expected value, the raw terms whose sum produced it:

    ORACLE = {..., "terms": {"CHECKSUM": [t0, t1, ...]}}

Two correct programs that add the same terms in a different order get slightly different
sums. The spread of the sum under reasonable reorderings (left to right, right to left,
tree order, blocks per worker, exactly rounded) is therefore the floor under any
meaningful tolerance. When terms are given, the spread is measured and the tolerance must
be at least TOL_MARGIN times the largest spread; a tolerance tighter than correct answers
can achieve is rejected.

From the oracle this module writes reference.json (language neutral, with the measured
spreads) and reference.h (C macros REF_<name> for every parameter and expected value, and
REF_TOL). Because the verification data comes from running the notebook itself, a change
to the notebook is followed by a change to the data the C code is tested against.

The notebook is run in a separate Python process with a timeout, so a notebook that
loops forever, crashes or exits cannot take the caller down with it.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import OracleError

log = logging.getLogger("parallelise.oracle")

TOL_MARGIN = 10.0  # tol must exceed the measured reordering spread by this factor
DEFAULT_TIMEOUT_S = 600.0

# The first line of reference.h. It is kept exactly as the earlier tool wrote it, so that
# regenerating a reference file leaves an unchanged oracle byte for byte unchanged.
HEADER_COMMENT = "/* generated from %s by make_oracle.py -- the oracle the C must match */"

# Runs in the child process. Arguments: notebook path, output path. Exit status 3 means a
# cell failed, 4 that ORACLE is absent, 5 that ORACLE cannot be written as JSON.
_RUNNER = r"""
import json, os, sys
path, out = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as handle:
    nb = json.load(handle)
namespace = {"__name__": "__main__"}
for i, cell in enumerate(nb.get("cells", [])):
    if cell.get("cell_type") != "code":
        continue
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    try:
        exec(compile(src, "%s cell %d" % (os.path.basename(path), i), "exec"), namespace)
    except BaseException as exc:
        sys.stderr.write("cell %d failed: %s: %s\n" % (i, type(exc).__name__, exc))
        sys.exit(3)
if "ORACLE" not in namespace:
    sys.stderr.write("no ORACLE\n")
    sys.exit(4)
def plain(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return float(o)
try:
    text = json.dumps(namespace["ORACLE"], default=plain)
except (TypeError, ValueError) as exc:
    sys.stderr.write("ORACLE cannot be written as JSON: %s\n" % exc)
    sys.exit(5)
with open(out, "w", encoding="utf-8") as handle:
    handle.write(text)
"""


@dataclass
class Oracle:
    """A validated oracle, plus where it came from and the measured reordering spreads."""

    kernel: str
    params: dict[str, int | float]
    expected: dict[str, int | float]
    tol: float
    terms: dict[str, list[float]] = field(default_factory=dict)
    source: str = ""
    reorder_spread: dict[str, float] = field(default_factory=dict)


# --- summation orders ---------------------------------------------------------------


def plain_sum(values: Sequence[float], lo: int, hi: int, step: int = 1) -> float:
    """Add values[lo:hi] one at a time, the way a C loop does (step -1 goes backwards).

    Python's built-in sum() has used compensated summation since version 3.12, so it is
    exactly rounded whatever the order and must not be used to measure ordering effects.
    """
    total = 0.0
    indices = range(lo, hi) if step > 0 else range(hi - 1, lo - 1, -1)
    for i in indices:
        total += values[i]
    return total


def pairwise_sum(values: Sequence[float], lo: int, hi: int) -> float:
    """Add in tree order, which is the shape of a parallel reduction."""
    if hi - lo <= 8:
        return plain_sum(values, lo, hi)
    mid = (lo + hi) // 2
    return pairwise_sum(values, lo, mid) + pairwise_sum(values, mid, hi)


def chunked_sum(values: Sequence[float], parts: int) -> float:
    """Add `parts` contiguous blocks separately (one per worker), then combine in order."""
    n = len(values)
    total = 0.0
    for r in range(parts):
        total += plain_sum(values, n * r // parts, n * (r + 1) // parts)
    return total


def reorder_spread(terms: Sequence[float]) -> float:
    """Return the largest disagreement between the sums of `terms` in the usual orders."""
    seq = [float(t) for t in terms]
    n = len(seq)
    candidates = [
        math.fsum(seq),  # exactly rounded, the anchor
        plain_sum(seq, 0, n),  # C loop, left to right
        plain_sum(seq, 0, n, step=-1),  # C loop, right to left
        pairwise_sum(seq, 0, n),  # tree order
    ]
    for parts in (2, 8, 64):  # blocks per worker, then combined
        if parts < n:
            candidates.append(chunked_sum(seq, parts))
    return max(candidates) - min(candidates)


# --- running the notebook ------------------------------------------------------------


def run_notebook(notebook: Path, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict[str, Any]:
    """Execute the notebook's code cells in a child process and return its ORACLE as data."""
    notebook = Path(notebook)
    if not notebook.is_file():
        raise OracleError(f"The notebook {notebook} does not exist. Check the path.")
    with tempfile.TemporaryDirectory(prefix="oracle-") as tmp:
        out = Path(tmp) / "oracle.json"
        command = [sys.executable, "-c", _RUNNER, str(notebook.resolve()), str(out)]
        try:
            proc = subprocess.run(
                command,
                cwd=notebook.resolve().parent,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                stdin=subprocess.DEVNULL,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise OracleError(
                f"Running {notebook.name} took longer than {timeout_s:g} seconds. "
                "Shorten the notebook's computation or allow more time."
            ) from None
        stderr = proc.stderr.strip()
        if proc.returncode == 3:
            raise OracleError(f"In {notebook.name}, {stderr}")
        if proc.returncode == 4:
            raise OracleError(
                f"{notebook.name} defines no ORACLE dictionary. "
                "Add a final cell that sets one."
            )
        if proc.returncode == 5:
            raise OracleError(f"In {notebook.name}, {stderr}")
        if proc.returncode != 0:
            raise OracleError(
                f"Running {notebook.name} stopped with exit status {proc.returncode}: "
                f"{stderr[-500:]}"
            )
        try:
            return {"ORACLE": json.loads(out.read_text(encoding="utf-8"))}
        except (OSError, ValueError) as exc:
            raise OracleError(
                f"The oracle from {notebook.name} could not be read back: {exc}"
            ) from exc


# --- validation ----------------------------------------------------------------------


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate(raw: object, source: str = "the notebook") -> Oracle:
    """Check the ORACLE dictionary against every rule and return it as an Oracle.

    Raises OracleError with a plain message naming the first rule that is broken,
    including the terms-sum check and the tolerance floor.
    """
    if not isinstance(raw, dict):
        raise OracleError("ORACLE must be a dictionary.")
    for key in ("kernel", "params", "expected", "tol"):
        if key not in raw:
            raise OracleError(f"ORACLE is missing the key {key!r}.")
    if not isinstance(raw["kernel"], str) or not raw["kernel"]:
        raise OracleError("ORACLE['kernel'] must be a non-empty string.")
    seen = {"TOL"}  # REF_TOL is written by this module, so no other name may clash with it
    for section in ("params", "expected"):
        entries = raw[section]
        if not isinstance(entries, dict) or not entries:
            raise OracleError(f"ORACLE[{section!r}] must be a non-empty dictionary.")
        for name, value in entries.items():
            if not isinstance(name, str) or not name.isidentifier() or not name.isascii():
                raise OracleError(
                    f"ORACLE[{section!r}] key {name!r} is not a valid C macro name."
                )
            if name in seen:
                raise OracleError(
                    f"The oracle name {name!r} is repeated; params, expected and TOL "
                    "must not share names."
                )
            seen.add(name)
            if not _is_number(value) or not math.isfinite(value):
                raise OracleError(f"ORACLE[{section!r}][{name!r}] must be a finite number.")
    tol = raw["tol"]
    if not _is_number(tol) or not math.isfinite(tol) or tol <= 0:
        raise OracleError("ORACLE['tol'] must be a finite positive number.")
    raw_terms = raw.get("terms", {})
    if not isinstance(raw_terms, dict):
        raise OracleError("ORACLE['terms'] must be a dictionary of name to list of numbers.")
    terms: dict[str, list[float]] = {}
    for name, seq in raw_terms.items():
        if name not in raw["expected"]:
            raise OracleError(f"ORACLE['terms'][{name!r}] has no matching expected value.")
        if not isinstance(seq, (list, tuple)) or not seq:
            raise OracleError(f"ORACLE['terms'][{name!r}] must be a non-empty list.")
        if not all(_is_number(t) and math.isfinite(t) for t in seq):
            raise OracleError(f"ORACLE['terms'][{name!r}] must contain only finite numbers.")
        terms[name] = [float(t) for t in seq]

    spreads: dict[str, float] = {}
    for name, seq in terms.items():
        total = math.fsum(seq)
        want = float(raw["expected"][name])
        if abs(total - want) > float(tol):
            raise OracleError(
                f"The terms for {name!r} add up to {total!r}, but the expected value is "
                f"{want!r}; the terms do not produce the expected value."
            )
        spreads[name] = reorder_spread(seq)
    if spreads:
        worst = max(spreads.values())
        floor = TOL_MARGIN * worst
        if float(tol) < floor:
            raise OracleError(
                f"The tolerance {float(tol):g} is below the floor {floor:g} (largest "
                f"reordering spread {worst:g} times the margin {TOL_MARGIN:g}). Correct "
                "programs that add in a different order would fail. Raise the tolerance."
            )
    return Oracle(
        kernel=raw["kernel"],
        params=dict(raw["params"]),
        expected=dict(raw["expected"]),
        tol=tol,
        terms=terms,
        source=source,
        reorder_spread=spreads,
    )


def extract_oracle(notebook: Path, timeout_s: float = DEFAULT_TIMEOUT_S) -> Oracle:
    """Run the notebook in a child process, then validate and return its oracle."""
    notebook = Path(notebook)
    namespace = run_notebook(notebook, timeout_s)
    oracle = validate(namespace["ORACLE"], source=notebook.name)
    log.info("extracted oracle for %r from %s", oracle.kernel, notebook)
    return oracle


# --- writing the reference files -------------------------------------------------------


def c_literal(value: int | float, force_double: bool = False) -> str:
    """Render a Python number as a C literal: an int becomes a long, a float a double.

    Floats are printed with repr, which gives the shortest text that reads back as exactly
    the same double.
    """
    if isinstance(value, bool):
        raise OracleError("Booleans have no place in an oracle.")
    if isinstance(value, int) and not force_double:
        return f"{value}L"
    text = repr(float(value))
    if not any(ch in text for ch in ".eE"):  # make sure C reads it as a double
        text += ".0"
    return text


def render_json(oracle: Oracle) -> str:
    """Return the text of reference.json."""
    record = {
        "source": oracle.source,
        "kernel": oracle.kernel,
        "params": oracle.params,
        "expected": oracle.expected,
        "tol": oracle.tol,
        "reorder_spread": oracle.reorder_spread,
    }
    return json.dumps(record, indent=2) + "\n"


def render_header(oracle: Oracle) -> str:
    """Return the text of reference.h."""
    lines = [HEADER_COMMENT % oracle.source, "#ifndef REFERENCE_H", "#define REFERENCE_H"]
    for name, value in oracle.params.items():
        lines.append(f"#define REF_{name} {c_literal(value)}")
    for name, value in oracle.expected.items():
        lines.append(f"#define REF_{name} {c_literal(value, force_double=True)}")
    lines.append(f"#define REF_TOL {c_literal(oracle.tol, force_double=True)}")
    lines.append("#endif")
    return "\n".join(lines) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    """Write through a uniquely named temporary file, so concurrent writers cannot collide."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_reference(oracle: Oracle, dest: Path) -> tuple[Path, Path]:
    """Write reference.json and reference.h into the folder `dest`; return both paths.

    Both files are rendered completely before either is written, so a failure cannot leave
    the pair from different generations.
    """
    dest = Path(dest)
    json_text = render_json(oracle)
    header_text = render_header(oracle)
    dest.mkdir(parents=True, exist_ok=True)
    json_path, header_path = dest / "reference.json", dest / "reference.h"
    _write_atomic(json_path, json_text)
    _write_atomic(header_path, header_text)
    return json_path, header_path
