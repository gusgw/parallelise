"""The study commands, `parallelise mask` and `parallelise report`.

`register` adds them to the main parser, so the command names and options are the same as
if they were defined in `parallelise.cli`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..cli_support import lazy_import, resolve_problem, settings_from_args
from ..errors import ParalleliseError
from ..results import ResultsStore


def cmd_mask(args: argparse.Namespace) -> int:
    """Write a copy of a problem with the algorithm's name and hints removed."""
    mask = lazy_import("study.mask")
    problem_dir = resolve_problem(args.problem, args.problems_dir)
    report = mask.mask_problem(problem_dir, Path(args.dest), args.alias)
    print(f"Masked {problem_dir.name} as {args.alias!r} in {args.dest}.")
    substitutions = getattr(report, "substitutions", None)
    if substitutions is not None:
        print(f"{len(substitutions)} substitutions made.")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Print the results tables and optionally draw the charts."""
    settings = settings_from_args(args)
    records = ResultsStore(settings.results_path).load()
    if not records:
        raise ParalleliseError(
            f"There are no run records in {settings.results_path}. Run 'parallelise run' "
            "first, or point --results at a results file."
        )
    report = lazy_import("study.report", "Install it with: pip install 'parallelise[report]'.")
    prices = report.load_prices(args.prices) if args.prices else None
    print(report.build_tables(records, prices=prices))
    if args.chart:
        report.build_chart(records, Path(args.chart))
        print(f"Chart written to {args.chart}.")
    if args.masked_chart:
        report.build_comparison_chart(records, Path(args.masked_chart))
        print(f"Masked-against-unmasked chart written to {args.masked_chart}.")
    return 0


def register(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    common: argparse.ArgumentParser,
    problems: argparse.ArgumentParser,
) -> None:
    """Add the mask and report commands to the main parser's subcommands.

    `common` holds the logging options and `problems` the --problems-dir option, which the
    main parser shares with its own commands.
    """
    p = sub.add_parser("mask", parents=[common, problems], help="write a masked copy of a problem")
    p.add_argument("problem")
    p.add_argument("alias")
    p.add_argument("dest")
    p.set_defaults(func=cmd_mask)

    p = sub.add_parser("report", parents=[common], help="print results tables")
    p.add_argument("--results", "--store", dest="results",
                   help="results file, also called the store (PARALLELISE_RESULTS)")
    p.add_argument("--prices",
                   help="JSON price table, US dollars per million tokens, for the cost column")
    p.add_argument("--chart", help="also write the pass-rate chart to this SVG file")
    p.add_argument("--masked-chart",
                   help="also write the masked-against-unmasked chart to this SVG file")
    p.set_defaults(func=cmd_report)
