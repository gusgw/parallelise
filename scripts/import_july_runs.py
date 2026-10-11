#!/usr/bin/env python3
"""Build results/runs.jsonl from the July 2026 run manifest and the saved evidence.

The July study recorded each graded run as one row of a tab-separated manifest, and kept
a short SUMMARY.txt per model and cell, plus the benchmark table and the kernel written
by each agent. This script turns those into one structured record per manifest row, in
the same shape that the package's ResultsStore uses, so that the report code treats the
July runs and later runs alike.

A cell is one combination of problem, backend, model and agent. A few cells were run
more than once; the evidence on disk belongs to the latest run only, so earlier rows
get no speed-up figure and only a coarse failure stage.

Usage, from the repository root:

    python scripts/import_july_runs.py            # writes results/runs.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path

log = logging.getLogger("parallelise.import_july_runs")

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "manifest-july-2026.tsv"
RUNS_DIR = ROOT / "results" / "runs"
OUTPUT = ROOT / "results" / "runs.jsonl"

# The field order matches RunRecord in src/parallelise/results.py.
FIELDS = (
    "when", "problem", "backend", "model", "agent", "agent_version", "masked",
    "result", "failure_stage", "speedup_best", "wall_time_s", "requests",
    "prompt_tokens", "completion_tokens", "latency_mean_s", "tokens_per_second",
    "serving", "quantisation", "notes",
)

# Serving facts for the July models, as recorded in docs/ARCHITECTURE.md.
BEDROCK = ("Amazon Bedrock through a LiteLLM gateway", "not disclosed")
SERVING = {
    "claude-sonnet-4.6": BEDROCK,
    "kimi-k2.5": BEDROCK,
    "llama-3.3-70b": BEDROCK,
    "gemma-31b": ("vLLM on one NVIDIA L40S with 48 GB",
                  "not recorded at the time (checkpoint and quantisation unknown)"),
}

STAGE_NOTES = {
    "no-files": "The agent did not write the kernel or its test.",
    "build": "The kernel or its test failed to build with warnings treated as errors.",
    "test-one-worker": "The unit test failed with a single worker.",
    "test-many-workers": "The unit test passed with one worker but failed with several.",
    "benchmark": "The benchmark failed or the speed-up was rejected.",
    "timeout": "The gate was stopped because it ran out of time.",
}

_TEST_ONE = re.compile(r"(OMP_NUM_THREADS=1 \./test_|-np 1 \./test_)")
_TEST_ANY = re.compile(r"^(OMP_NUM_THREADS=\S+ |mpirun .*)?\./test_\w+\s*$")


def split_summary(text: str) -> dict[str, str]:
    """Split a SUMMARY.txt into its per-agent sections, keyed by agent name."""
    sections: dict[str, str] = {}
    for part in re.split(r"^======== ", text, flags=re.M)[1:]:
        name = part.split(" ", 1)[0]
        sections[name] = part
    return sections


def classify_section(section: str) -> tuple[str, str]:
    """Return (result, failure_stage) for one agent section of a SUMMARY.txt."""
    if "GATE PASS" in section and "GATE FAIL" not in section:
        return "PASS", "none"
    lines = section.splitlines()
    if "Terminated" in section or "timed out" in section:
        return "FAIL", "timeout"
    if any("cannot access" in ln or "No rule to make target" in ln for ln in lines):
        return "FAIL", "no-files"
    test_runs = [ln for ln in lines if _TEST_ANY.match(ln.strip())]
    if any(ln.startswith("./bench") for ln in lines):
        return "FAIL", "benchmark"
    if len(test_runs) >= 2:
        return "FAIL", "test-many-workers"
    if len(test_runs) == 1:
        return "FAIL", "test-one-worker"
    return "FAIL", "build"


def best_speedup(csv_path: Path) -> float | None:
    """Return the best speed-up in a benchmark table, or None if there is none."""
    if not csv_path.is_file():
        return None
    best: float | None = None
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if "impl" in row and row["impl"] != "gpu":
                continue
            try:
                value = float(row["speedup"])
            except (KeyError, TypeError, ValueError):
                continue
            best = value if best is None else max(best, value)
    return best


def build_records(manifest: Path = MANIFEST, runs_dir: Path = RUNS_DIR) -> list[dict]:
    """Build one record per manifest row, oldest first."""
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    latest: dict[tuple, int] = {}
    for index, row in enumerate(rows):
        key = (row["problem"], row["backend"], row["model"], row["agent"])
        latest[key] = index  # the manifest is in time order, so the last index wins
    records = []
    for index, row in enumerate(rows):
        key = (row["problem"], row["backend"], row["model"], row["agent"])
        problem, backend, model, agent = key
        serving, quantisation = SERVING.get(model, ("not recorded", "not disclosed"))
        notes: list[str] = []
        stage = "none" if row["result"] == "PASS" else "benchmark"
        speedup = None
        if latest[key] == index:
            cell = runs_dir / problem / backend / model
            summary = cell / "SUMMARY.txt"
            section = None
            if summary.is_file():
                section = split_summary(summary.read_text()).get(agent)
            if section is None:
                notes.append("No SUMMARY.txt section was kept for this run.")
            else:
                result, derived = classify_section(section)
                if result != row["result"]:
                    notes.append(f"The manifest says {row['result']} but the summary "
                                 f"says {result}; the manifest is used.")
                elif result == "FAIL":
                    stage = derived
                    notes.append(STAGE_NOTES[derived])
            if row["result"] == "PASS" or stage in ("benchmark", "timeout"):
                speedup = best_speedup(cell / agent / "bench.csv")
        else:
            notes.append("A later run of the same cell replaced this one, so no "
                         "per-run evidence is kept.")
            if row["result"] == "FAIL":
                notes.append("The failure stage is not known.")
        records.append({
            "when": row["when"], "problem": problem, "backend": backend,
            "model": model, "agent": agent, "agent_version": row["agent_md5"],
            "masked": False, "result": row["result"], "failure_stage": stage,
            "speedup_best": speedup, "wall_time_s": None, "requests": None,
            "prompt_tokens": None, "completion_tokens": None,
            "latency_mean_s": None, "tokens_per_second": None,
            "serving": serving, "quantisation": quantisation,
            "notes": " ".join(notes),
        })
    return records


def main(argv: list[str] | None = None) -> int:
    """Write results/runs.jsonl and report how many records were written."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    records = build_records()
    for record in records:
        assert tuple(record) == FIELDS
    with args.output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    log.info("wrote %d records to %s", len(records), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
