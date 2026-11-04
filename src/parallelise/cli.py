"""The command-line interface: `parallelise <subcommand>`.

Exit status is 0 on success, 1 when a check fails or an error is reported, and 2 for bad
arguments. Every error message says what to do next.

The commands listed by `parallelise --help` are health, assess, run, check and study. The
study command holds the tools for studying many runs: `study mask` and `study report`. Three
maintenance commands (oracle, assemble and bench-check) work but are not listed, and the
older names grade, mask and report remain as hidden aliases that print a notice. The modules
that do the heavy work (check, assemble, agents, assess and the study tools) are imported
only when the subcommand that needs them runs. A missing or incomplete module therefore
produces a clear message for that subcommand alone, instead of stopping the whole program at
start-up.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import logging
import shutil
import sys
import time
from pathlib import Path

from . import __version__
from .cli_support import PROBLEM_HELP, HelpParser
from .cli_support import lazy_import as _lazy
from .cli_support import resolve_problem as _resolve_problem
from .cli_support import settings_from_args as _settings
from .errors import AgentUnavailable, ConfigError, ParalleliseError
from .log import configure_logging
from .model import ModelClient
from .oracle import extract_oracle, write_reference
from .results import ResultsStore, RunRecord
from .study import cli as study_cli

log = logging.getLogger("parallelise.cli")

BACKENDS = ("openmp", "mpi", "cuda")
AGENTS = ("basic", "pydantic", "langgraph")

_BACKEND_ADVICE = {
    "openmp": (
        "Use OpenMP (shared-memory threads). The unit test is plain C and the routine is "
        "built with -fopenmp."
    ),
    "mpi": (
        "Use MPI (message passing between processes). The unit test is run under mpirun at "
        "one and at several processes; only rank 0 may report results."
    ),
    "cuda": (
        "Use CUDA (a graphics processor). The routine is a .cu file built with nvcc, and the "
        "check requires a real speed-up over one processor core."
    ),
}


def build_task(name: str, backend: str) -> str:
    """Return the instructions given to the agent for a problem called `name`."""
    routine_file = name + (".cu" if backend == "cuda" else ".c")
    return (
        "You convert a research Jupyter notebook that implements a numerical algorithm in "
        f"serial Python into a parallel C routine. Read the notebook with the notebook tool to "
        f"learn the algorithm. Read {name}.h, which gives the exact signature to implement, "
        "and reference.h, which gives the expected values and the tolerance REF_TOL. If "
        "problem.h is present, read it and use its shared helpers. Then write two files: "
        f"{routine_file}, the parallel routine implementing {name}.h, and test_{name}.c, a unit "
        "test that checks the routine against reference.h within REF_TOL. The benchmark "
        "bench.c is already provided; do not write or change it. Divide the work among the "
        "parallel workers so that each does only its share, and combine the partial results "
        "correctly; never make every worker repeat the whole computation. The build treats "
        "warnings as errors, so include every header whose functions you call and leave no "
        "parameter unused. " + _BACKEND_ADVICE[backend] + " Build and test with 'make check' "
        "and keep fixing until it passes. Copy text exactly from what you read, and make "
        "small, exact edits. Finish only when the checks pass."
    )


def _agent_version(agent: object) -> str:
    """Return a short hash of the agent's source file, identifying the exact design tested."""
    try:
        source = inspect.getsourcefile(type(agent))
        if source:
            return hashlib.sha256(Path(source).read_bytes()).hexdigest()[:10]
    except (OSError, TypeError):
        pass
    return "unknown"


def _safe(text: str) -> str:
    """Make a model name safe to use as a folder name."""
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text) or "model"


def _check_backend(backend: str) -> None:
    if backend not in BACKENDS:
        raise ParalleliseError(f"The backend {backend!r} is not one of {', '.join(BACKENDS)}.")


# --- subcommands -------------------------------------------------------------------


def cmd_health(args: argparse.Namespace) -> int:
    """Check that the model endpoint is reachable and serves the chosen model."""
    settings = _settings(args)
    with ModelClient(settings) as client:
        report = client.health_check()
    print(f"Endpoint: {settings.gateway_url}")
    print(f"Model requested: {settings.model}")
    print(f"Reachable: {'yes' if report.reachable else 'no'}")
    if report.reachable:
        print(f"Models offered: {', '.join(report.models) or 'none'}")
        print(f"Requested model listed: {'yes' if report.model_listed else 'no'}")
        print(f"Test chat call succeeded: {'yes' if report.chat_ok else 'no'}")
        print(f"Round trip: {report.round_trip_s:.2f} seconds")
    if report.detail:
        print(report.detail, file=sys.stderr)
    return 0 if report.healthy else 1


def cmd_oracle(args: argparse.Namespace) -> int:
    """Run a notebook and write reference.json and reference.h."""
    notebook = Path(args.notebook)
    oracle = extract_oracle(notebook, timeout_s=args.timeout)
    dest = Path(args.out) if args.out else notebook.resolve().parent
    json_path, header_path = write_reference(oracle, dest)
    print(f"Wrote {json_path} and {header_path} from the notebook {notebook.name}.")
    return 0


def cmd_assemble(args: argparse.Namespace) -> int:
    """Build a work folder for a problem and backend."""
    _check_backend(args.backend)
    assemble = _lazy("assemble")
    problem_dir = _resolve_problem(args.problem, args.problems_dir)
    assemble.assemble_work_dir(
        problem_dir, args.backend, Path(args.dest), include_reference=args.with_reference
    )
    print(f"Assembled {problem_dir.name} for {args.backend} in {args.dest}.")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Run the check on a finished work folder."""
    _check_backend(args.backend)
    work_dir = Path(args.work_dir)
    if not work_dir.is_dir():
        raise ParalleliseError(f"The work folder {work_dir} does not exist. Check the path.")
    check = _lazy("check")
    result = check.run_check(
        work_dir, args.backend, timeout_s=int(args.timeout), speedup_floor=args.floor
    )
    print(result.log)
    speedup = "none" if result.speedup_best is None else f"{result.speedup_best:.2f}"
    print(f"Check {'PASSED' if result.passed else 'FAILED'} "
          f"(failure stage: {result.failure_stage}, best speed-up: {speedup}).")
    return 0 if result.passed else 1


def cmd_grade(args: argparse.Namespace) -> int:
    """Run the check under its old name, after a one-line notice."""
    print("parallelise: the command 'grade' is now called 'check'; running 'check'.",
          file=sys.stderr)
    return cmd_check(args)


def cmd_bench_check(args: argparse.Namespace) -> int:
    """Validate the speed-up figures in a benchmark CSV file."""
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise ParalleliseError(f"The benchmark file {csv_path} does not exist. Check the path.")
    bench = _lazy("bench_check")
    verdict = bench.check_cpu_table(csv_path, args.floor)
    for reason in verdict.reasons:
        print(f"- {reason}")
    best = "none" if verdict.speedup_best is None else f"{verdict.speedup_best:.2f}"
    print(f"Benchmark {'accepted' if verdict.ok else 'rejected'}; best speed-up {best}.")
    return 0 if verdict.ok else 1


def cmd_assess(args: argparse.Namespace) -> int:
    """Ask the model which backend suits a notebook; print the analysis and the verdict."""
    assess = _lazy("assess")
    notebook = Path(args.notebook)
    settings = _settings(args)
    with ModelClient(settings) as client:
        verdict = assess.assess_notebook(notebook, client)
    print(verdict.analysis)
    print()
    print(f"{verdict.summary_line()} (model {verdict.model}).")
    if args.out:
        assess.write_verdict(verdict, Path(args.out))
        print(f"Verdict written to {args.out}.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Assemble a work folder, run an agent in it, check the result and record the run."""
    _check_backend(args.backend)
    settings = _settings(args)
    assemble = _lazy("assemble")
    check = _lazy("check")
    agents = _lazy("agents")
    problem_dir = _resolve_problem(args.problem, args.problems_dir)
    problem_name = problem_dir.name
    work_root = Path(settings.work_root)

    name = problem_name
    masked = bool(args.masked)
    if masked:
        mask = _lazy("study.mask")
        masked_dir = work_root / "_masked" / problem_name
        if masked_dir.exists():
            shutil.rmtree(masked_dir)
        mask.mask_problem(problem_dir, masked_dir, args.masked)
        problem_dir, name = masked_dir, args.masked

    backend = args.backend

    agent = agents.get_agent(args.agent)  # may raise AgentUnavailable with an install hint
    work_dir = work_root / problem_name / backend / _safe(settings.model) / args.agent
    if work_dir.exists():
        shutil.rmtree(work_dir)
    assemble.assemble_work_dir(problem_dir, backend, work_dir, include_reference=False)

    task = build_task(name, backend)
    started = time.perf_counter()
    result = None
    agent_error = ""
    with ModelClient(settings) as client:
        try:
            result = agent.run(work_dir, task, settings, client, step_budget=args.step_budget)
        except AgentUnavailable:
            raise
        except Exception as exc:  # an agent crash is a result to record, not a reason to stop
            log.exception("the agent failed")
            agent_error = f"{type(exc).__name__}: {exc}"
        summary = client.summary()

    if agent_error:
        passed, stage, speedup, check_log = False, "agent-error", None, ""
    else:
        graded = check.run_check(work_dir, backend, timeout_s=int(args.check_timeout))
        passed, stage, speedup, check_log = (
            graded.passed,
            "none" if graded.passed else graded.failure_stage,
            graded.speedup_best,
            graded.log,
        )
        (work_dir / "check.log").write_text(check_log, encoding="utf-8")
    wall = time.perf_counter() - started

    notes: list[str] = []
    if args.notes:
        notes.append(args.notes)
    if agent_error:
        reason = str(agent_error).rstrip(".")
        notes.append(f"The agent stopped with {reason}.")
    if result is not None:
        if result.notes:
            notes.append(result.notes)
        if not result.finished:
            notes.append("The agent used its step budget without declaring itself finished.")
    latencies = list(result.latencies_s) if result is not None else []
    # Prefer the agent's own counts; fall back to the client's when the agent kept none.
    requests = result.requests if result and result.requests else summary["requests"]
    prompt = result.prompt_tokens if result and result.prompt_tokens else summary["prompt_tokens"]
    completion = (
        result.completion_tokens if result and result.completion_tokens
        else summary["completion_tokens"]
    )
    waiting = sum(latencies) if latencies else 0.0
    record = RunRecord(
        problem=problem_name,
        backend=backend,
        model=settings.model,
        agent=args.agent,
        agent_version=_agent_version(agent),
        masked=masked,
        result="PASS" if passed else "FAIL",
        failure_stage=stage,
        speedup_best=speedup,
        wall_time_s=round(wall, 2),
        requests=requests,
        prompt_tokens=prompt,
        completion_tokens=completion,
        latency_mean_s=(waiting / len(latencies)) if latencies else summary["mean_latency_s"],
        tokens_per_second=(completion / waiting) if waiting > 0
        else summary["completion_tokens_per_second"],
        serving=args.serving,
        quantisation=args.quantisation,
        notes=" ".join(notes),
    )
    ResultsStore(settings.results_path).append(record)
    print(f"{record.result}: {problem_name} on {backend} with {args.agent} and "
          f"{settings.model} (failure stage: {stage}). Record added to {settings.results_path}.")
    return 0 if passed else 1


# --- argument parsing ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = HelpParser(
        prog="parallelise",
        description="Turn a serial Python notebook into a verified parallel C program.",
    )
    parser.add_argument("--version", action="version", version=f"parallelise {__version__}")
    common = HelpParser(add_help=False)
    common.add_argument("--log-level", default="WARNING", help="DEBUG, INFO, WARNING or ERROR")
    common.add_argument("--log-json", action="store_true", help="write log lines as JSON")
    endpoint = HelpParser(add_help=False)
    endpoint.add_argument("--gateway-url", help="model endpoint (PARALLELISE_GATEWAY_URL)")
    endpoint.add_argument("--model", help="model name (PARALLELISE_MODEL)")
    problems = HelpParser(add_help=False)
    problems.add_argument("--problems-dir", default="problems",
                          help="folder holding the shipped examples (default: problems)")

    sub = parser.add_subparsers(
        dest="command", required=True, metavar="command", parser_class=HelpParser
    )

    p = sub.add_parser(
        "health", parents=[common, endpoint],
        help="check that the model endpoint answers and serves the chosen model",
        description="Check that the model endpoint answers, lists the chosen model and "
                    "completes one small chat request.",
        epilog="Example: parallelise health --gateway-url http://localhost:4000/v1 --model local",
    )
    p.set_defaults(func=cmd_health)

    p = sub.add_parser(
        "assess", parents=[common, endpoint],
        help="ask the model which parallel method suits a notebook",
        description="Ask the model which parallel method suits a notebook, or whether plain "
                    "serial code is better, and print its reasoning and verdict.",
        epilog="Example: parallelise assess problems/integrate/integrate.ipynb --out verdict.json",
    )
    p.add_argument("notebook", help="the path to the Jupyter notebook to assess")
    p.add_argument("--out", help="write the verdict, as JSON, to this file")
    p.set_defaults(func=cmd_assess)

    p = sub.add_parser(
        "run", parents=[common, endpoint, problems],
        help="convert a problem with an agent, check the result and record the run",
        description="Build a work folder for a problem, let an agent write the parallel "
                    "program in it, check the result and add a record to the results file.",
        epilog="Example: parallelise run integrate openmp --agent basic --model local",
    )
    p.add_argument("problem", help=PROBLEM_HELP)
    p.add_argument("backend", choices=BACKENDS,
                   help="the parallel method to use: openmp, mpi or cuda")
    p.add_argument("--agent", choices=AGENTS, default="basic",
                   help="the design of agent that drives the model (default: basic)")
    p.add_argument("--masked", metavar="ALIAS",
                   help="hide the algorithm's name behind this neutral alias, such as problem_a")
    p.add_argument("--step-budget", type=int, default=80, help="most agent steps allowed")
    p.add_argument("--check-timeout", type=float, default=300.0,
                   help="seconds allowed for the check")
    p.add_argument("--work-root", help="folder for work folders (PARALLELISE_WORK_ROOT)")
    p.add_argument("--results", help="results file (PARALLELISE_RESULTS)")
    p.add_argument("--serving", default="not recorded", help="who served the model")
    p.add_argument("--quantisation", default="not disclosed", help="model precision, if known")
    p.add_argument("--notes", default="", help="free text stored with the record")
    p.set_defaults(func=cmd_run)

    def add_check_arguments(parser_: argparse.ArgumentParser) -> None:
        parser_.add_argument(
            "work_dir",
            help="the path to a finished work folder (not a problem), such as one made by "
                 "'parallelise run'")
        parser_.add_argument("--backend", required=True, choices=BACKENDS,
                             help="the parallel method the work folder was built for")
        parser_.add_argument("--timeout", type=float, default=300.0,
                             help="seconds allowed for the whole check (default: 300)")
        parser_.add_argument("--floor", type=float, default=None,
                             help="the least best speed-up to accept; may only tighten the "
                                  "floor set in the problem's Makefile")

    p = sub.add_parser(
        "check", parents=[common],
        help="judge a finished work folder again, without the agent or the model",
        description="Judge a finished work folder: its trusted files, its build, its test at "
                    "one worker and at several, and its benchmark. The argument is a work "
                    "folder, not a problem.",
        epilog="Example: parallelise check work/integrate/openmp/local/basic --backend openmp",
    )
    add_check_arguments(p)
    p.set_defaults(func=cmd_check)

    # A subcommand added without a help argument is not listed by "parallelise --help".
    p = sub.add_parser("grade", parents=[common],
                       description="The old name of 'check'.")
    add_check_arguments(p)
    p.set_defaults(func=cmd_grade)

    study_cli.register(sub, common, problems)  # the study command and its old aliases

    p = sub.add_parser(
        "oracle", parents=[common],
        description="Run a notebook and write the reference files (reference.json and "
                    "reference.h) from the ORACLE dictionary in its last cell.",
        epilog="Example: parallelise oracle problems/integrate/integrate.ipynb",
    )
    p.add_argument("notebook", help="the path to the Jupyter notebook to run")
    p.add_argument("--out", help="folder for the reference files (default: the notebook's)")
    p.add_argument("--timeout", type=float, default=600.0, help="seconds allowed for the notebook")
    p.set_defaults(func=cmd_oracle)

    p = sub.add_parser(
        "assemble", parents=[common, problems],
        description="Build a work folder for a problem and a backend, without running an agent.",
        epilog="Example: parallelise assemble integrate openmp work/try --with-reference",
    )
    p.add_argument("problem", help=PROBLEM_HELP)
    p.add_argument("backend", choices=BACKENDS, help="the parallel method: openmp, mpi or cuda")
    p.add_argument("dest", help="the folder to create; it is emptied first if it exists")
    p.add_argument("--with-reference", action="store_true",
                   help="also copy the known-good solution (tests the machinery without a model)")
    p.set_defaults(func=cmd_assemble)

    p = sub.add_parser(
        "bench-check", parents=[common],
        description="Apply the benchmark table rules to a CSV file of OpenMP or MPI timings.",
        epilog="Example: parallelise bench-check work/integrate/openmp/local/basic/bench.csv",
    )
    p.add_argument("csv", help="the path to the benchmark table to judge")
    p.add_argument("--floor", type=float, default=0.0,
                   help="the least best speed-up to accept (default 0: no floor)")
    p.set_defaults(func=cmd_bench_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line and return the exit status."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits itself; turn that into a return value
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 2)
    try:
        configure_logging(args.log_level, args.log_json)
        return int(args.func(args))
    except ConfigError as exc:
        print(f"parallelise: configuration problem. {exc}", file=sys.stderr)
        return 1
    except ParalleliseError as exc:
        print(f"parallelise: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("parallelise: interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
