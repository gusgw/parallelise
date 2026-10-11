"""CLI:  python -m eval_harness [--harness NAME ...] [--task KEY ...] [--out FILE].

Runs each selected harness over each selected task against the real targets, so it
needs the pipeline targets present (their deps installed) and a reachable model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from eval_harness.harnesses import HARNESSES
from eval_harness.runner import (
    best_of_n, render_bestofn, render_markdown, run_eval, write_artifacts)
from eval_harness.tasks import TASKS

DEFAULT_ROOT = str(Path(__file__).resolve().parents[3])  # the harness/ directory


def main() -> None:
    parser = argparse.ArgumentParser(prog="eval_harness")
    parser.add_argument("--harness", action="append", help="harness name (default: all)")
    parser.add_argument("--task", action="append", help="task key (default: all)")
    parser.add_argument("--out", default=None, help="write markdown here (default: stdout)")
    parser.add_argument("--harness-root", default=DEFAULT_ROOT)
    parser.add_argument("--n", type=int, default=1,
                        help="attempts per (harness, task); >1 = best-of-N in isolated worktrees")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="max concurrent attempts when --n > 1")
    args = parser.parse_args()

    harness_list = [HARNESSES[n] for n in (args.harness or list(HARNESSES))]
    task_list = [t for t in TASKS if not args.task or t.key in args.task]

    if args.n > 1:
        results = [best_of_n(h, t, args.harness_root, n=args.n, concurrency=args.concurrency)
                   for t in task_list for h in harness_list]
        md = render_bestofn(results)
        artifacts = (args.out + ".jsonl") if args.out else "artifacts.jsonl"
        write_artifacts(results, artifacts)
        print("wrote per-attempt artifacts: " + artifacts)
    else:
        results = run_eval(harness_list, task_list, args.harness_root)
        md = render_markdown(results)

    if args.out:
        Path(args.out).write_text(md + "\n")
        print("wrote " + args.out)
    else:
        print(md)


if __name__ == "__main__":
    main()
