"""The command-line interface: `parallelise <subcommand>`.

Exit status is 0 on success, 1 when a check fails or an error is reported, and 2 for bad
arguments. Every error message says what to do next.

The modules that do the heavy work (check, assemble, mask, agents, report) are imported
only when the subcommand that needs them runs. A missing or incomplete module therefore
produces a clear message for that subcommand alone, instead of stopping the whole program
at start-up.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import logging
import shutil
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

from . import __version__
from .config import Settings
from .errors import AgentUnavailable, ConfigError, ParalleliseError
from .log import configure_logging
from .model import ModelClient
from .oracle import extract_oracle, write_reference
from .results import ResultsStore, RunRecord

log = logging.getLogger("parallelise.cli")

BACKENDS = ("openmp", "mpi", "cuda")
AGENTS = ("basic", "pydantic", "langgraph")

_BACKEND_ADVICE = {
    "openmp": (
        "Use OpenMP (shared-memory threads). The unit test is plain C and the kernel is "
        "built with -fopenmp."
    ),
    "mpi": (
        "Use MPI (message passing between processes). The unit test is run under mpirun at "
        "one and at several processes; only rank 0 may report results."
    ),
    "cuda": (
        "Use CUDA (a graphics processor). The kernel is a .cu file built with nvcc, and the "
        "check requires a real speed-up over one processor core."
    ),
}


def _lazy(name: str, hint: str = "") -> ModuleType:
    """Import parallelise.<name> on first use, turning an import failure into a plain error."""
    full = f"parallelise.{name}"
    try:
        return importlib.import_module(full)
    except ModuleNotFoundError as exc:
        if exc.name == full:
            raise ParalleliseError(
                f"The {name!r} part of parallelise is not available in this installation. "
                "Reinstall the package with 'pip install -e .' from the repository root."
            ) from exc
        raise ParalleliseError(
            f"The {name!r} part of parallelise needs the Python package {exc.name!r}, which is "
            f"not installed. {hint or 'Install it with pip.'}"
        ) from exc
    except ImportError as exc:
        raise ParalleliseError(
            f"The {name!r} part of parallelise could not be loaded ({exc}). "
            "Reinstall the package with 'pip install -e .'."
        ) from exc


def build_task(name: str, backend: str) -> str:
    """Return the instructions given to the agent for a problem called `name`."""
    kernel_file = name + (".cu" if backend == "cuda" else ".c")
    return (
        "You convert a research Jupyter notebook that implements a numerical algorithm in "
        f"serial Python into a parallel C kernel. Read the notebook with the notebook tool to "
        f"learn the algorithm. Read {name}.h, which gives the exact signature to implement, "
        "and reference.h, which gives the expected values and the tolerance REF_TOL. If "
        "problem.h is present, read it and use its shared helpers. Then write two files: "
        f"{kernel_file}, the parallel kernel implementing {name}.h, and test_{name}.c, a unit "
        "test that checks the kernel against reference.h within REF_TOL. The benchmark "
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


def _resolve_problem(problem: str, problems_dir: str) -> Path:
    """Accept either a path to a problem folder or the name of one under problems_dir."""
    direct = Path(problem)
    if direct.is_dir():
        return direct
    named = Path(problems_dir) / problem
    if named.is_dir():
        return named
    raise ParalleliseError(
        f"There is no problem folder {problem!r} or {str(named)!r}. Run from the repository "
        "root, or give the folder's path, or set --problems-dir."
    )


def _check_backend(backend: str, allow_auto: bool = False) -> None:
    allowed = BACKENDS + (("auto",) if allow_auto else ())
    if backend not in allowed:
        raise ParalleliseError(f"The backend {backend!r} is not one of {', '.join(allowed)}.")


def _problem_notebook(problem_dir: Path) -> Path:
    """Return the notebook of a problem folder."""
    named = problem_dir / f"{problem_dir.name}.ipynb"
    if named.is_file():
        return named
    found = sorted(problem_dir.glob("*.ipynb"))
    if len(found) == 1:
        return found[0]
    raise ParalleliseError(
        f"The problem folder {problem_dir} has no single notebook to assess. "
        "Add one named after the folder."
    )


def _settings(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {
        "gateway_url": getattr(args, "gateway_url", None),
        "model": getattr(args, "model", None),
        "work_root": getattr(args, "work_root", None),
        "results_path": getattr(args, "results", None),
    }
    return Settings.from_env(overrides)


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
    print(f"Wrote {json_path} and {header_path} for kernel {oracle.kernel!r}.")
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


def cmd_grade(args: argparse.Namespace) -> int:
    """Run the check on an existing work folder."""
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


def cmd_mask(args: argparse.Namespace) -> int:
    """Write a copy of a problem with the algorithm's name and hints removed."""
    mask = _lazy("mask")
    problem_dir = _resolve_problem(args.problem, args.problems_dir)
    report = mask.mask_problem(problem_dir, Path(args.dest), args.alias)
    print(f"Masked {problem_dir.name} as {args.alias!r} in {args.dest}.")
    substitutions = getattr(report, "substitutions", None)
    if substitutions is not None:
        print(f"{len(substitutions)} substitutions made.")
    return 0


def cmd_bench_check(args: argparse.Namespace) -> int:
    """Validate the speed-up figures in a benchmark CSV file."""
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise ParalleliseError(f"The benchmark file {csv_path} does not exist. Check the path.")
    bench = _lazy("bench_check")
    verdict = bench.check_bench_csv(csv_path, args.floor)
    for reason in verdict.reasons:
        print(f"- {reason}")
    best = "none" if verdict.speedup_best is None else f"{verdict.speedup_best:.2f}"
    print(f"Benchmark {'accepted' if verdict.ok else 'rejected'}; best speed-up {best}.")
    return 0 if verdict.ok else 1


def cmd_report(args: argparse.Namespace) -> int:
    """Print the results tables and optionally draw the chart."""
    settings = _settings(args)
    records = ResultsStore(settings.results_path).load()
    if not records:
        raise ParalleliseError(
            f"There are no run records in {settings.results_path}. Run 'parallelise run' "
            "first, or point --results at a results file."
        )
    report = _lazy("report", "Install it with: pip install 'parallelise[report]'.")
    prices = report.load_prices(args.prices) if args.prices else None
    print(report.build_tables(records, prices=prices))
    if args.chart:
        report.build_chart(records, Path(args.chart))
        print(f"Chart written to {args.chart}.")
    if args.masked_chart:
        report.build_comparison_chart(records, Path(args.masked_chart))
        print(f"Masked-against-unmasked chart written to {args.masked_chart}.")
    return 0


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


def _choose_backend(problem_dir: Path, settings: Settings) -> tuple[str, Any]:
    """Assess the problem's notebook and return the backend to run with the verdict.

    Raises ParalleliseError, which the command line reports with exit status 1, when the
    recommendation is serial code or names a backend the problem does not have.
    """
    assess = _lazy("assess")
    with ModelClient(settings) as client:
        verdict = assess.assess_notebook(_problem_notebook(problem_dir), client)
    chosen = verdict.chosen_backend
    print(f"Assessment: {verdict.summary_line()}")
    if chosen == "serial":
        raise ParalleliseError(
            f"The assessment recommends well-optimised serial code for {problem_dir.name}, "
            "so no parallel run was made and no record was written. To run a backend anyway, "
            "name it instead of 'auto'. Justification: " + (verdict.justification or "none given")
        )
    if not (problem_dir / chosen).is_dir():
        have = ", ".join(b for b in BACKENDS if (problem_dir / b).is_dir()) or "none"
        raise ParalleliseError(
            f"The assessment recommends {chosen} for {problem_dir.name}, but the problem has "
            f"no {chosen} folder (it has: {have}), so no run was made and no record was "
            "written. Name one of those backends instead of 'auto'."
        )
    return chosen, verdict


def cmd_run(args: argparse.Namespace) -> int:
    """Assemble a work folder, run an agent in it, grade the result and record the run.

    With the backend "auto" the model is first asked which backend suits the notebook.
    """
    _check_backend(args.backend, allow_auto=True)
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
        mask = _lazy("mask")
        masked_dir = work_root / "_masked" / problem_name
        if masked_dir.exists():
            shutil.rmtree(masked_dir)
        mask.mask_problem(problem_dir, masked_dir, args.masked)
        problem_dir, name = masked_dir, args.masked

    backend = args.backend
    verdict = None
    if backend == "auto":
        backend, verdict = _choose_backend(problem_dir, settings)

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
    if verdict is not None:
        _lazy("assess").write_verdict(verdict, work_dir / "assessment.json")
        notes.append(
            f"The backend {backend} was chosen by the assessment ({verdict.model}); the "
            "verdict is saved as assessment.json in the work folder, and its "
            f"{verdict.prompt_tokens} prompt and {verdict.completion_tokens} completion tokens "
            "are not counted in this record."
        )
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
    parser = argparse.ArgumentParser(
        prog="parallelise",
        description="Turn a serial Python notebook into a verified parallel C kernel.",
    )
    parser.add_argument("--version", action="version", version=f"parallelise {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default="WARNING", help="DEBUG, INFO, WARNING or ERROR")
    common.add_argument("--log-json", action="store_true", help="write log lines as JSON")
    endpoint = argparse.ArgumentParser(add_help=False)
    endpoint.add_argument("--gateway-url", help="model endpoint (PARALLELISE_GATEWAY_URL)")
    endpoint.add_argument("--model", help="model name (PARALLELISE_MODEL)")
    problems = argparse.ArgumentParser(add_help=False)
    problems.add_argument("--problems-dir", default="problems", help="folder of problems")

    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("health", parents=[common, endpoint], help="check the model endpoint")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("oracle", parents=[common], help="run a notebook, write reference files")
    p.add_argument("notebook")
    p.add_argument("--out", help="folder for the reference files (default: the notebook's)")
    p.add_argument("--timeout", type=float, default=600.0, help="seconds allowed for the notebook")
    p.set_defaults(func=cmd_oracle)

    p = sub.add_parser("assemble", parents=[common, problems], help="build a work folder")
    p.add_argument("problem")
    p.add_argument("backend", choices=BACKENDS)
    p.add_argument("dest")
    p.add_argument("--with-reference", action="store_true",
                   help="also copy the known-good solution (tests the machinery without a model)")
    p.set_defaults(func=cmd_assemble)

    p = sub.add_parser("run", parents=[common, endpoint, problems],
                       help="run an agent on a problem and record the graded result")
    p.add_argument("problem")
    p.add_argument("backend", choices=BACKENDS + ("auto",),
                   help="a backend, or auto to ask the model first and use its recommendation")
    p.add_argument("--agent", choices=AGENTS, default="basic")
    p.add_argument("--masked", metavar="ALIAS",
                   help="hide the algorithm's name behind this neutral alias")
    p.add_argument("--step-budget", type=int, default=80, help="most agent steps allowed")
    p.add_argument("--check-timeout", type=float, default=300.0, help="seconds allowed for grading")
    p.add_argument("--work-root", help="folder for work folders (PARALLELISE_WORK_ROOT)")
    p.add_argument("--results", help="results file (PARALLELISE_RESULTS)")
    p.add_argument("--serving", default="not recorded", help="who served the model")
    p.add_argument("--quantisation", default="not disclosed", help="model precision, if known")
    p.add_argument("--notes", default="", help="free text stored with the record")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("assess", parents=[common, endpoint],
                       help="ask the model which backend suits a notebook")
    p.add_argument("notebook")
    p.add_argument("--out", help="write the verdict, as JSON, to this file")
    p.set_defaults(func=cmd_assess)

    p = sub.add_parser("grade", parents=[common], help="run the check on a work folder")
    p.add_argument("work_dir")
    p.add_argument("--backend", required=True, choices=BACKENDS)
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--floor", type=float, default=None, help="minimum best speed-up")
    p.set_defaults(func=cmd_grade)

    p = sub.add_parser("mask", parents=[common, problems], help="write a masked copy of a problem")
    p.add_argument("problem")
    p.add_argument("alias")
    p.add_argument("dest")
    p.set_defaults(func=cmd_mask)

    p = sub.add_parser("bench-check", parents=[common], help="validate a benchmark CSV file")
    p.add_argument("csv")
    p.add_argument("--floor", type=float, default=0.0,
                   help="minimum best speed-up (default 0: no floor)")
    p.set_defaults(func=cmd_bench_check)

    p = sub.add_parser("report", parents=[common], help="print results tables")
    p.add_argument("--results", "--store", dest="results",
                   help="results file, also called the store (PARALLELISE_RESULTS)")
    p.add_argument("--prices",
                   help="JSON price table, US dollars per million tokens, for the cost column")
    p.add_argument("--chart", help="also write the pass-rate chart to this SVG file")
    p.add_argument("--masked-chart",
                   help="also write the masked-against-unmasked chart to this SVG file")
    p.set_defaults(func=cmd_report)
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
