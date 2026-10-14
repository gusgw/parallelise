"""Tables and a chart that summarise the graded runs.

The study runs each combination of problem, backend, model and agent design (a "cell")
and sometimes runs a cell again. To keep a repeated cell from counting twice, every
table and the chart use only the latest run of each cell, judged by its timestamp.
Runs on masked problems (where the algorithm's name was hidden) are a separate
experiment, so they are left out unless the caller asks for them.

Records may be `RunRecord` objects or plain dictionaries with the same field names.

When a store holds both masked and unmasked runs, `build_tables` also compares the two.
When the records carry request counts, it adds a serving table (latency, speed, tokens and,
if the caller supplies a price table, cost).
"""
from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("parallelise.study.report")

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


def _number(record: Any, name: str) -> float:
    """Read a numeric field, treating a missing field as zero."""
    try:
        return float(_get(record, name) or 0)
    except (KeyError, AttributeError):
        return 0.0


def load_prices(path: str | Path) -> dict[str, tuple[float, float]]:
    """Read a price table: US dollars per million tokens, as (input, output) for each model.

    The file is JSON. Each model name maps to an object with `input` and `output`, for
    example {"some-model": {"input": 3.0, "output": 15.0}}. Entries whose value is not an
    object, such as a "source" note, are ignored.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    table = data.get("usd_per_million_tokens", data)
    return {name: (float(entry["input"]), float(entry["output"]))
            for name, entry in table.items() if isinstance(entry, dict)}


def _change(unmasked: tuple[int, int], masked: tuple[int, int]) -> str:
    """Format the change in pass rate from unmasked to masked, in percentage points."""
    if not unmasked[1] or not masked[1]:
        return "-"
    delta = 100 * masked[0] / masked[1] - 100 * unmasked[0] / unmasked[1]
    rounded = round(delta)
    return f"{rounded:+d}" if rounded else "0"


def _comparison_rows(plain: Sequence[Any], masked: Sequence[Any],
                     keys: Sequence[tuple[str, dict[str, str]]]) -> list[list[str]]:
    """Rows of (label, unmasked count, masked count, change) for each labelled field filter."""
    def pick(cells: Sequence[Any], want: dict[str, str]) -> list[Any]:
        return [c for c in cells if all(_get(c, k) == v for k, v in want.items())]

    rows = []
    for label, want in keys:
        before = _count(pick(plain, want))
        after = _count(pick(masked, want))
        rows.append([label, _fmt(before), _fmt(after), _change(before, after)])
    return rows


def build_comparison_tables(records: Iterable[Any]) -> str:
    """Return Markdown tables comparing masked with unmasked runs, or "" if one is absent."""
    records = list(records)
    plain = latest_cells(records)
    masked = latest_cells(records, masked=True)
    if not plain or not masked:
        return ""
    models = _model_order(plain)
    pairs = sorted({(_get(c, "problem"), _get(c, "backend")) for c in plain + masked},
                   key=lambda pb: (pb[0], _backend_rank(pb[1])))
    header = ["Unmasked", "Masked", "Change (points)"]
    by_model = [(m, {"model": m}) for m in models] + [("All models", {})]
    by_pair = [(f"{p} on {b}", {"problem": p, "backend": b}) for p, b in pairs]
    note = ("Passes out of runs, for the same problems with and without the masking. "
            "Change is masked minus unmasked pass rate, in percentage points.")
    return ("### Masked against unmasked, by model\n\n" + note + "\n\n"
            + _markdown_table(["Model"] + header, _comparison_rows(plain, masked, by_model))
            + "\n\n### Masked against unmasked, by problem and backend\n\n" + note + "\n\n"
            + _markdown_table(["Problem and backend"] + header,
                              _comparison_rows(plain, masked, by_pair)) + "\n")


def _backend_rank(backend: str) -> tuple[int, str]:
    """Sort key that puts known backends first, in their usual order."""
    return (BACKEND_ORDER.index(backend) if backend in BACKEND_ORDER else len(BACKEND_ORDER),
            backend)


def build_serving_table(records: Iterable[Any],
                        prices: dict[str, tuple[float, float]] | None = None) -> str:
    """Return a Markdown table of requests, latency, speed, tokens and cost by model.

    It uses the latest run of each cell, masked and unmasked together. Runs that recorded no
    requests (for example a run that stopped on an error before its counts were saved) are
    left out of the medians. The cost is tokens times the price per million tokens that
    the caller passes in; a model without a price shows a dash. Returns "" when no run
    recorded any request.
    """
    records = list(records)
    cells = latest_cells(records) + latest_cells(records, masked=True)
    if not any(_number(c, "requests") for c in cells):
        return ""
    prices = prices or {}
    header = ["Model", "Runs", "Requests", "Median latency per call (s)",
              "Median completion tokens per second", "Prompt tokens", "Completion tokens",
              "Agent time (hours)"]
    if prices:
        header.append("Cost (US dollars)")
    rows = []
    totals = [0.0] * 5
    total_cost = 0.0
    for model in _model_order(cells):
        mine = [c for c in cells if _get(c, "model") == model]
        timed = [c for c in mine if _number(c, "requests") and _number(c, "latency_mean_s")]
        speed = [c for c in mine if _number(c, "tokens_per_second")]
        sums = [float(len(mine))] + [sum(_number(c, f) for c in mine) for f in
                ("requests", "prompt_tokens", "completion_tokens", "wall_time_s")]
        row = [model, f"{len(mine)}", f"{sums[1]:,.0f}",
               f"{statistics.median(_number(c, 'latency_mean_s') for c in timed):.2f}"
               if timed else "-",
               f"{statistics.median(_number(c, 'tokens_per_second') for c in speed):.0f}"
               if speed else "-",
               f"{sums[2]:,.0f}", f"{sums[3]:,.0f}", f"{sums[4] / 3600:.1f}"]
        for i in range(5):
            totals[i] += sums[i]
        if prices:
            if model in prices:
                cost = (sums[2] * prices[model][0] + sums[3] * prices[model][1]) / 1e6
                total_cost += cost
                row.append(f"{cost:,.2f}")
            else:
                row.append("-")
        rows.append(row)
    total_row = ["All models", f"{totals[0]:.0f}", f"{totals[1]:,.0f}", "", "",
                 f"{totals[2]:,.0f}", f"{totals[3]:,.0f}", f"{totals[4] / 3600:.1f}"]
    if prices:
        total_row.append(f"{total_cost:,.2f}")
    rows.append(total_row)
    note = ("Totals over the latest run of every cell, masked and unmasked together. "
            "Medians are over runs. Agent time is the sum of the wall time of the runs.")
    if prices:
        note += (" Cost uses the price table supplied, in US dollars per million tokens: "
                 + "; ".join(f"{m} {i:g} in and {o:g} out"
                             for m, (i, o) in sorted(prices.items())) + ".")
    return "### Serving: requests, speed, tokens and cost by model\n\n" + note + "\n\n" \
        + _markdown_table(header, rows) + "\n"


def build_tables(records: Iterable[Any], masked: bool = False,
                 prices: dict[str, tuple[float, float]] | None = None) -> str:
    """Return Markdown tables of pass counts by model and backend, agent and problem.

    Only the latest run of each cell is counted; each caption says so. When the records
    hold both masked and unmasked runs, the masked-against-unmasked tables follow. When they
    hold request counts, a serving table follows, with cost if `prices` maps a model name
    to (input, output) US dollars per million tokens.
    """
    records = list(records)
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
    text = "\n\n".join(parts) + "\n"
    for extra in (build_comparison_tables(records), build_serving_table(records, prices)):
        if extra:
            text += "\n" + extra
    return text


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
        ax.set_title("Share of cells passing the check, by model and backend", fontsize=11,
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


def build_comparison_chart(records: Iterable[Any], dest_svg: Path) -> Path:
    """Draw grouped bars of pass rate by model, unmasked beside masked, saved as SVG.

    Only the latest run of each cell is counted. The file is written to `dest_svg`, whose
    folder is created if needed, and the path is returned.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = list(records)
    plain = latest_cells(records)
    masked = latest_cells(records, masked=True)
    dest = Path(dest_svg)
    dest.parent.mkdir(parents=True, exist_ok=True)
    models = _model_order(plain or masked)
    series = (("Unmasked", plain, "#0072B2"), ("Masked", masked, "#E69F00"))

    with plt.rc_context({"svg.hashsalt": "parallelise", "svg.fonttype": "none",
                         "font.family": "sans-serif"}):
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        width = 0.8 / len(series)
        for j, (name, cells, colour) in enumerate(series):
            for i, model in enumerate(models):
                passed, total = _count(c for c in cells if _get(c, "model") == model)
                if not total:
                    continue
                x = i + (j - (len(series) - 1) / 2) * width
                ax.bar(x, 100 * passed / total, width * 0.92, color=colour,
                       label=name if i == 0 else None)
                ax.text(x, 100 * passed / total + 1.5, f"{passed}/{total}",
                        ha="center", va="bottom", fontsize=7.5, color="#333333")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=9)
        ax.set_ylim(0, 112)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("Pass rate (%)")
        ax.set_title("Share of runs passing the check, unmasked and masked", fontsize=11,
                     loc="left")
        ax.yaxis.grid(True, color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(frameon=False, ncol=len(series), loc="upper right", fontsize=8.5)
        fig.text(0.01, 0.01, "Labels show passes/runs. Latest run of each cell only.",
                 fontsize=7.5, color="#555555")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(dest, format="svg", metadata={"Date": None})
        plt.close(fig)
    log.info("wrote chart to %s", dest)
    return dest
