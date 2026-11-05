"""The study commands, `parallelise study mask` and `parallelise study report`.

`register` adds the `study` command, with those two subcommands, to the main parser. It also
adds the older top-level names `mask` and `report` as hidden aliases, which print a one-line
notice and then run the new form.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..cli_support import PROBLEM_HELP, HelpParser, lazy_import, resolve_problem, settings_from_args
from ..errors import ParalleliseError
from ..results import ResultsStore


def cmd_mask(args: argparse.Namespace) -> int:
    """Write a copy of a problem with the algorithm's name and hints removed."""
    mask = lazy_import("study.mask")
    problem_dir = resolve_problem(args.problem, args.problems_dir)
    report = mask.mask_problem(problem_dir, Path(args.dest), args.alias, force=args.force)
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
    reference = report.load_reference(args.reference) if args.reference else None
    print(report.build_tables(records, prices=prices, reference=reference))
    if args.chart:
        _draw(report.build_chart, records, args.chart)
        print(f"Chart written to {args.chart}.")
    if args.masked_chart:
        _draw(report.build_comparison_chart, records, args.masked_chart)
        print(f"Masked-against-unmasked chart written to {args.masked_chart}.")
    return 0


def _draw(build: Callable[..., Path], records: list[Any], dest: str) -> None:
    """Draw one chart, turning a missing library or an unwritable file into a plain error."""
    try:
        build(records, Path(dest))
    except ImportError as exc:
        raise ParalleliseError(
            "The charts need the Python package matplotlib, which is not installed. Install "
            "it with: pip install 'parallelise[report]'."
        ) from exc
    except OSError as exc:
        raise ParalleliseError(
            f"The chart {dest} could not be written ({exc.strerror or exc}). Check the path and "
            "its permissions."
        ) from exc


def _old_name(new_form: str, func: Callable[[argparse.Namespace], int]) -> Callable[
    [argparse.Namespace], int
]:
    """Wrap a command so that it first prints a notice pointing at its new form."""

    def run(args: argparse.Namespace) -> int:
        print(f"parallelise: this command is now called '{new_form}'; running it.",
              file=sys.stderr)
        return func(args)

    return run


def _add_mask(sub: argparse._SubParsersAction[HelpParser],
              name: str, parents: list[HelpParser], hidden: bool,
              func: Callable[[argparse.Namespace], int]) -> None:
    # Without a help argument a subcommand is not listed in its parent's help.
    extra: dict[str, Any] = {}
    if not hidden:
        extra["help"] = "write a copy of a problem with the name hidden"
    p = sub.add_parser(
        name, parents=parents,
        description="Write a copy of a problem with the algorithm's name and hints removed, "
                    "so that a model cannot recognise a textbook algorithm by its name.",
        epilog="Example: parallelise study mask scan problem_c masked/problem_c",
        **extra,
    )
    p.add_argument("problem", help=PROBLEM_HELP)
    p.add_argument("alias", help="the neutral name to use instead, such as problem_a")
    p.add_argument("dest", help="the folder to write the masked copy into; it must not exist "
                                "or must be empty, unless --force is given")
    p.add_argument("--force", action="store_true",
                   help="delete what is in the destination folder first, instead of refusing")
    p.set_defaults(func=func)


def _add_report(sub: argparse._SubParsersAction[HelpParser],
                name: str, parents: list[HelpParser], hidden: bool,
                func: Callable[[argparse.Namespace], int]) -> None:
    extra: dict[str, Any] = {}
    if not hidden:
        extra["help"] = "print the results tables, and optionally charts"
    p = sub.add_parser(
        name, parents=parents,
        description="Read the results file and print the pass counts, costs and speeds as "
                    "Markdown tables. A pass is a correct routine, whatever its speed. The "
                    "speed table counts the correct runs by speed verdict, with their median "
                    "best speed-up, by problem, backend and model; with --reference it shows "
                    "the known-good solution's speed-up beside them. It takes no positional "
                    "argument.",
        epilog="Example: parallelise study report --results results/local-runs.jsonl "
               "--reference reference-speedups.json --chart pass-rate.svg",
        **extra,
    )
    p.add_argument("--results", "--store", dest="results",
                   help="the results file, also called the store (PARALLELISE_RESULTS)")
    p.add_argument("--prices",
                   help="JSON price table, US dollars per million tokens, for the cost column")
    p.add_argument("--reference", metavar="FILE",
                   help="JSON list of objects with problem, backend and best_speedup: the "
                        "known-good solution's best speed-up on the machine, shown beside "
                        "each model's speed")
    p.add_argument("--chart", help="also write the pass-rate chart to this SVG file")
    p.add_argument("--masked-chart",
                   help="also write the masked-against-unmasked chart to this SVG file")
    p.set_defaults(func=func)


def register(
    sub: argparse._SubParsersAction[HelpParser],
    common: HelpParser,
    problems: HelpParser,
) -> None:
    """Add the study command, and the hidden old names of its subcommands, to the main parser.

    `common` holds the logging options and `problems` the --problems-dir option, which the
    main parser shares with its own commands.
    """
    study = sub.add_parser(
        "study", parents=[common],
        help="tools for studying many runs: mask a problem, report on the results",
        description="Tools for studying many runs together, apart from converting one "
                    "problem: 'mask' hides the algorithm's name, and 'report' summarises "
                    "the results file.",
    )
    study_sub = study.add_subparsers(
        dest="study_command", required=True, metavar="tool", parser_class=HelpParser
    )
    _add_mask(study_sub, "mask", [common, problems], False, cmd_mask)
    _add_report(study_sub, "report", [common], False, cmd_report)

    _add_mask(sub, "mask", [common, problems], True,
              _old_name("parallelise study mask", cmd_mask))
    _add_report(sub, "report", [common], True,
                _old_name("parallelise study report", cmd_report))
