"""Tables, and a chart, that summarise the runs in a results file.

Every run counts, repeats included. A pass means that the routine was correct; its speed is
reported apart, for the correct runs only. There are three tables:

- correct runs for each model on each backend, with a 95 per cent Wilson interval (a range
  for the true pass rate that stays sound for small counts);
- the speed of the correct runs for each problem, backend and model: how many were faster
  than the serial routine, and their median speed-up and scaling;
- the use of each model: requests, the median latency of a call, the median rate of
  completion tokens, the tokens used, the agent's time and, with a price table, the cost.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from ..errors import ParalleliseError
from ..generate import BACKENDS
from ..results import RunRecord


def wilson(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson interval for a pass rate, as two fractions."""
    if total == 0:
        return 0.0, 1.0
    rate = passed / total
    scale = 1 + z * z / total
    centre = (rate + z * z / (2 * total)) / scale
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / scale
    return max(0.0, centre - half), min(1.0, centre + half)


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |"]  # fmt: skip
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _models(records: Sequence[RunRecord]) -> list[str]:
    return sorted({r.model for r in records})


def correctness_table(records: Sequence[RunRecord]) -> str:
    """Correct runs out of all runs, for each model on each backend, with Wilson intervals."""
    backends = [b for b in BACKENDS if any(r.backend == b for r in records)]
    rows = []
    for model in _models(records):
        row = [model]
        for backend in backends:
            cell = [r for r in records if r.model == model and r.backend == backend]
            passed = sum(r.correct for r in cell)
            low, high = wilson(passed, len(cell))
            shown = f"{passed}/{len(cell)} ({100 * low:.0f} to {100 * high:.0f}%)"
            row.append(shown if cell else "")
        rows.append(row)
    return _table(["Model", *backends], rows)


def speed_table(records: Sequence[RunRecord]) -> str:
    """For the correct runs of each problem, backend and model: how many were faster than the
    serial routine, and their median speed-up and scaling."""
    groups: dict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
    for r in records:
        if r.correct and r.speedup is not None:
            groups[(r.problem, r.backend, r.model)].append(r)
    rows = []
    for (problem, backend, model), runs in sorted(groups.items()):
        scalings = [r.scaling for r in runs if r.scaling is not None]
        rows.append([
            problem, backend, model, str(len(runs)),
            str(sum(r.faster for r in runs)),
            f"{statistics.median(r.speedup or 0.0 for r in runs):.2f}",
            f"{statistics.median(scalings):.2f}" if scalings else "",
        ])  # fmt: skip
    header = ["Problem", "Backend", "Model", "Correct runs", "Faster", "Median speed-up",
              "Median scaling"]  # fmt: skip
    return _table(header, rows)


def model_table(records: Sequence[RunRecord],
                prices: dict[str, tuple[float, float]] | None = None) -> str:  # fmt: skip
    """The use of each model, with its cost when a price table is given."""
    header = ["Model", "Runs", "Requests", "Median latency of a call (s)",
              "Median completion tokens per second", "Prompt tokens", "Completion tokens",
              "Agent time (hours)"]  # fmt: skip
    if prices:
        header.append("Cost (US dollars)")
    rows = []
    for model in _models(records):
        runs = [r for r in records if r.model == model]
        prompt = sum(r.prompt_tokens for r in runs)
        completion = sum(r.completion_tokens for r in runs)
        row = [model, str(len(runs)), str(sum(r.requests for r in runs)),
               f"{statistics.median(r.latency_mean_s for r in runs):.2f}",
               f"{statistics.median(r.tokens_per_second for r in runs):.0f}",
               str(prompt), str(completion),
               f"{sum(r.agent_time_s or 0.0 for r in runs) / 3600:.1f}"]  # fmt: skip
        if prices:
            price_in, price_out = prices.get(model, (math.nan, math.nan))
            row.append(f"{(prompt * price_in + completion * price_out) / 1e6:.2f}")
        rows.append(row)
    return _table(header, rows)


def build_tables(records: Sequence[RunRecord],
                 prices: dict[str, tuple[float, float]] | None = None) -> str:  # fmt: skip
    """All three tables, each under a heading."""
    if not records:
        return "There are no runs to report."
    unjudged = sum(r.failure_stage == "notebook" for r in records)
    judged = [r for r in records if r.failure_stage != "notebook"]
    parts = ["Correct runs (every judged run counts; 95 per cent Wilson interval):",
             correctness_table(judged)]  # fmt: skip
    if unjudged:
        parts.append(f"{unjudged} {'run is' if unjudged == 1 else 'runs are'} left out, "
                     "because the notebook's own Python failed during the judgement.")
    speed = speed_table(records)
    if speed.count("\n") > 1:
        parts += ["Speed of the correct runs against the serial routine:", speed]
    parts += ["Use of each model:", model_table(records, prices)]
    return "\n\n".join(parts)


def load_prices(path: str | Path) -> dict[str, tuple[float, float]]:
    """Read a price table: US dollars per million tokens, as (input, output) for each model.

    The file is JSON that maps each model to an object with "input" and "output", at its top
    level or under "usd_per_million_tokens"; entries that are not objects, such as a note of
    the source, are ignored. Raises ParalleliseError, naming the file, if it cannot be used.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        table = data.get("usd_per_million_tokens", data)
        return {name: (float(entry["input"]), float(entry["output"]))
                for name, entry in table.items() if isinstance(entry, dict)}  # fmt: skip
    except (OSError, UnicodeDecodeError, ValueError, AttributeError, KeyError, TypeError) as exc:
        raise ParalleliseError(
            f"The price file {path} cannot be used ({exc}). It must be JSON that maps each "
            'model to an object with "input" and "output" prices in US dollars per million '
            "tokens."
        ) from exc


def build_chart(records: Sequence[RunRecord], dest_svg: Path) -> Path:
    """Draw the share of correct runs for each model and backend, as an SVG bar chart."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ParalleliseError("The chart needs matplotlib, which is not installed. Install it "
                               "with: pip install 'parallelise[report]'.") from exc

    records = [r for r in records if r.failure_stage != "notebook"]  # as in the table
    models = _models(records)
    backends = [b for b in BACKENDS if any(r.backend == b for r in records)]
    figure, axes = plt.subplots(figsize=(max(4.0, 1.6 * len(models)), 3.2))
    width = 0.8 / max(len(backends), 1)
    for i, backend in enumerate(backends):
        shares = []
        for model in models:
            cell = [r for r in records if r.model == model and r.backend == backend]
            shares.append(sum(r.correct for r in cell) / len(cell) if cell else 0.0)
        axes.bar([m + i * width for m in range(len(models))], shares, width, label=backend)
    axes.set_xticks([m + width * (len(backends) - 1) / 2 for m in range(len(models))], models)
    axes.set_ylim(0, 1)
    axes.set_ylabel("share of runs correct")
    axes.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(dest_svg, format="svg")
    plt.close(figure)
    return Path(dest_svg)
