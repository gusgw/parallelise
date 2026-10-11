"""Draw the temperature-sweep chart from the per-attempt result files.

Each point is the fraction of 20 attempts at the debugging task that passed, for the
31B model. The numbers are read from the RESULTS jsonl files, not typed in, so the
chart cannot drift from the data. Run it from this folder:

    python make_temperature_chart.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EVAL = Path(__file__).resolve().parent.parent / "eval"
TASK = "debug-best-weather"
FILES = {
    0.0: "RESULTS-temp-0.0.md.jsonl",
    0.3: "RESULTS-temp-0.3.md.jsonl",
    0.7: "RESULTS-31b.md.jsonl",
    1.0: "RESULTS-temp-1.0.md.jsonl",
}
STRUCTURED = ("agent_pydantic_typed", "agent_pydantic_typed_checked")
LINES = [  # agent, label, colour, marker
    ("agent_langgraph", "LangGraph with deepagents", "#2a78d6", "o"),
    ("agent_langgraph_small", "LangGraph, small tool set", "#1baf7a", "s"),
    ("agent_pydantic", "Pydantic AI, with verification", "#eb6834", "^"),
    ("agent_pydantic_min", "Pydantic AI, no forcing", "#e87ba4", "v"),
    ("agent_min_improved", "Hand-written loop, with gates", "#4a3aa7", "D"),
    ("agent_min", "Hand-written loop (control)", "#52514e", "P"),
]


def rates() -> dict[str, dict[float, float]]:
    """Return the pass rate per agent and temperature."""
    out: dict[str, dict[float, float]] = defaultdict(dict)
    for temp, name in FILES.items():
        wins: dict[str, int] = defaultdict(int)
        total: dict[str, int] = defaultdict(int)
        for line in (EVAL / name).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec["task"] != TASK:
                continue
            total[rec["harness"]] += 1
            wins[rec["harness"]] += bool(rec["passed"])
        for agent, n in total.items():
            out[agent][temp] = 100.0 * wins[agent] / n
    return out


def main() -> None:
    data = rates()
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=150)
    for agent, label, colour, marker in LINES:
        pts = sorted(data[agent].items())
        ax.plot([t for t, _ in pts], [r for _, r in pts], marker=marker, color=colour,
                lw=2, ms=7, label=label, mec="white", mew=1)
    zero = sorted(data[STRUCTURED[0]].items())
    ax.plot([t for t, _ in zero], [r for _, r in zero], ls="--", color="#8a8985", lw=2,
            label="Structured-finish agents (both)")
    ax.set_xlabel("Sampling temperature")
    ax.set_ylabel("Attempts that passed (%)")
    ax.set_title("Success rate against sampling temperature (31B model, 20 attempts each)",
                 fontsize=10.5, loc="left")
    ax.set_xticks([0.0, 0.3, 0.7, 1.0])
    ax.set_ylim(-4, 104)
    ax.set_xlim(-0.04, 1.04)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(Path(__file__).with_name("temperature-sweep.svg"), format="svg")


if __name__ == "__main__":
    main()
