"""Tables and a chart that summarise the graded runs.

The study runs each combination of problem, backend, model and agent design (a "cell")
and sometimes runs a cell again. To keep a repeated cell from counting twice, every
table and the chart use only the latest run of each cell, judged by its timestamp.
Runs on masked problems (where the algorithm's name was hidden) are a separate
experiment, so they are left out unless the caller asks for them.

Records may be `RunRecord` objects or plain dictionaries with the same field names.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("parallelise.report")

BACKEND_ORDER = ("openmp", "mpi", "cuda")
CAPTION = ("Counts are passes out of cells. A cell is one problem, backend, model and agent "
           "design; only the latest run of each cell is counted.")

# A palette that stays distinguishable for the common kinds of colour-blindness.
_BACKEND_COLOURS = {"openmp": "#0072B2", "mpi": "#E69F00", "cuda": "#009E73"}
_EXTRA_COLOURS = ("#CC79A7", "#56B4E9", "#D55E00")


def _get(record: Any, name: str) -> Any:
    """Read a field from a record that is either a dictionary or an object."""
    if isinstance(record, dict):
        return record[name]
    return getattr(record, name)


def _when(record: Any) -> datetime:
    """Parse a record's timestamp; a missing time zone is taken to be UTC."""
    stamp = datetime.fromisoformat(str(_get(record, "when")))
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=UTC)


def latest_cells(records: Iterable[Any], masked: bool = False) -> list[Any]:
    """Return the latest run of each cell, for masked or unmasked problems only.

    Ties on the timestamp go to the record that appears later in the input.
    """
    best: dict[tuple, tuple[datetime, int, Any]] = {}
    for position, record in enumerate(records):
        if bool(_get(record, "masked")) != masked:
            continue
        key = tuple(_get(record, f) for f in ("problem", "backend", "model", "agent"))
        stamp = _when(record)
        if key not in best or (stamp, position) >= best[key][:2]:
            best[key] = (stamp, position, record)
    return [entry[2] for _, entry in sorted(best.items())]


def _ordered_backends(cells: Sequence[Any]) -> list[str]:
    """Known backends in their usual order, then any others alphabetically."""
    present = {_get(c, "backend") for c in cells}
    known = [b for b in BACKEND_ORDER if b in present]
    return known + sorted(present - set(BACKEND_ORDER))


def _count(cells: Iterable[Any]) -> tuple[int, int]:
    """Return (passes, total) for some cells."""
    passed = total = 0
    for cell in cells:
        total += 1
        passed += _get(cell, "result") == "PASS"
    return passed, total


def _fmt(passed_total: tuple[int, int]) -> str:
    """Format a count as `passes/total (percentage)`, or a dash when empty."""
    passed, total = passed_total
    if total == 0:
        return "-"
    return f"{passed}/{total} ({100 * passed / total:.0f}%)"


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table with a left-aligned first column."""
    lines = ["| " + " | ".join(header) + " |",
             "| --- |" + " ---: |" * (len(header) - 1)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _grouped(cells: Sequence[Any], row_field: str, col_field: str,
             rows: Sequence[str], cols: Sequence[str]) -> list[list[str]]:
    """Build table rows of pass counts by one field (rows) and another (columns)."""
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for cell in cells:
        groups[(_get(cell, row_field), _get(cell, col_field))].append(cell)
    out = []
    for row in rows:
        row_cells = [c for c in cells if _get(c, row_field) == row]
        out.append([row] + [_fmt(_count(groups[(row, col)])) for col in cols]
                   + [_fmt(_count(row_cells))])
    return out


def _model_order(cells: Sequence[Any]) -> list[str]:
    """Models from best to worst overall pass rate, ties broken by name."""
    def rate(model: str) -> float:
        passed, total = _count(c for c in cells if _get(c, "model") == model)
        return passed / total if total else 0.0
    return sorted({_get(c, "model") for c in cells}, key=lambda m: (-rate(m), m))


def build_tables(records: Iterable[Any], masked: bool = False) -> str:
    """Return Markdown tables of pass counts by model and backend, agent and problem.

    Only the latest run of each cell is counted; each caption says so.
    """
    cells = latest_cells(records, masked=masked)
    if not cells:
        return "No runs to report.\n"
    backends = _ordered_backends(cells)
    models = _model_order(cells)
    agents = sorted({_get(c, "agent") for c in cells})
    problems = sorted({_get(c, "problem") for c in cells})
    parts = [
        "### Passes by model and backend\n\n" + CAPTION + "\n\n"
        + _markdown_table(["Model"] + backends + ["All"],
                          _grouped(cells, "model", "backend", models, backends)),
        "### Passes by agent design and model\n\n" + CAPTION + "\n\n"
        + _markdown_table(["Agent design"] + models + ["All"],
                          _grouped(cells, "agent", "model", agents, models)),
        "### Passes by problem and model\n\n" + CAPTION + "\n\n"
        + _markdown_table(["Problem"] + models + ["All"],
                          _grouped(cells, "problem", "model", problems, models)),
    ]
    return "\n\n".join(parts) + "\n"


def build_chart(records: Iterable[Any], dest_svg: Path, masked: bool = False) -> Path:
    """Draw a grouped bar chart of pass rate by model and backend, saved as SVG.

    Only the latest run of each cell is counted. The file is written to `dest_svg`,
    whose folder is created if needed, and the path is returned.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = latest_cells(records, masked=masked)
    dest = Path(dest_svg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    backends = _ordered_backends(cells)
    models = _model_order(cells)
    colours = {b: _BACKEND_COLOURS.get(b) or _EXTRA_COLOURS[i % len(_EXTRA_COLOURS)]
               for i, b in enumerate(backends)}

    # Fixed hash salt and text-as-text keep the SVG identical from run to run.
    with plt.rc_context({"svg.hashsalt": "parallelise", "svg.fonttype": "none",
                         "font.family": "sans-serif"}):
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        width = 0.8 / max(len(backends), 1)
        labelled: set[str] = set()
        for j, backend in enumerate(backends):
            for i, model in enumerate(models):
                group = [c for c in cells
                         if _get(c, "model") == model and _get(c, "backend") == backend]
                passed, total = _count(group)
                if not total:
                    continue
                x = i + (j - (len(backends) - 1) / 2) * width
                ax.bar(x, 100 * passed / total, width * 0.92, color=colours[backend],
                       label=None if backend in labelled else backend)
                labelled.add(backend)
                ax.text(x, 100 * passed / total + 1.5, f"{passed}/{total}",
                        ha="center", va="bottom", fontsize=7.5, color="#333333")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=9)
        ax.set_ylim(0, 112)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("Pass rate (%)")
        ax.set_title("Share of cells passing the gate, by model and backend", fontsize=11,
                     loc="left")
        ax.yaxis.grid(True, color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(title="Backend", frameon=False, ncol=len(backends), loc="upper right",
                  fontsize=8.5, title_fontsize=8.5)
        fig.text(0.01, 0.01, "Labels show passes/cells. Latest run of each cell only.",
                 fontsize=7.5, color="#555555")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(dest, format="svg", metadata={"Date": None})
        plt.close(fig)
    log.info("wrote chart to %s", dest)
    return dest
