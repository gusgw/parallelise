"""The command line: parallelise serial, run, check, reverse and report.

An error that a person can cause is printed as one line that says what went wrong and what to
do next, and the exit status is then 1.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path

from . import __version__, convert, generate, judge, notebook, provenance, reverse, serial
from .agents import AGENT_NAMES, AgentResult, get_agent
from .config import Settings
from .errors import AgentUnavailable, ParalleliseError
from .log import configure_logging
from .model import ModelClient
from .results import ResultsStore, RunRecord
from .study import report

log = logging.getLogger("parallelise.cli")

#: What each failure stage means, for the closing line of run and check.
_STAGE_WORDS = {
    "no-files": "the routine file is missing",
    "rule": "the routine breaks the rule on what a routine may contain",
    "build": "the routine did not compile, with warnings treated as errors",
    "test": "the routine gave a wrong answer, or stopped, at a test size",
    "benchmark": "the routine gave a wrong answer, or stopped, at the benchmark size",
    "timeout": "the judgement ran out of time",
}


def verdict_line(verdict: judge.Verdict) -> str:
    """Return one plain line that gives a judgement's correctness and speed together."""
    if verdict.stage == "notebook":
        return "NOT JUDGED: the notebook's own Python failed during the judgement (stage notebook)"
    if not verdict.correct:
        what = _STAGE_WORDS.get(verdict.stage, "the judgement failed")
        return f"FAIL: {what} (stage {verdict.stage}); speed not measured"
    shown = "" if verdict.speedup is None else f" (speed-up {verdict.speedup:.2f})"
    faster = "faster" if verdict.faster else "not faster"
    return f"PASS: correct; {faster} than the serial routine{shown}"


def _settings(args: argparse.Namespace) -> Settings:
    """The settings from the options that were given and from the environment variables."""
    return Settings.from_env({
        "gateway_url": getattr(args, "gateway_url", None), "model": getattr(args, "model", None),
        "work_root": getattr(args, "work_root", None),
        "results_path": getattr(args, "results", None),
        "time_limit_s": getattr(args, "time_limit", None),
    })  # fmt: skip


def _problem(folder: str) -> tuple[Path, Path]:
    """Return a problem folder and its notebook, which has the folder's name."""
    path = Path(folder)
    nb = path / f"{path.name}.ipynb"
    if not nb.is_file():
        raise ParalleliseError(f"{path} is not a problem folder: it holds no {nb.name}. Give the "
                               "path of a folder that holds a notebook of the same name.")
    return path, nb


def _require_tools(backend: str) -> None:
    """Raise ParalleliseError, before anything takes long, if the judgement could not run."""
    missing = judge.missing_tools(backend)
    if missing:
        raise ParalleliseError(f"Judging a routine for {backend} needs {', '.join(missing)}, "
                               "which this machine lacks. Install them, or work on a machine "
                               "that has them.")  # fmt: skip


def _new_work_folder(parent: Path) -> Path:
    """Make a new folder under ``parent``, named by the time, so that no run meets another's."""
    parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H-%M-%SZ", time.gmtime())
    for n in range(1, 100):
        folder = parent / (stamp if n == 1 else f"{stamp}-{n}")
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            continue
    raise ParalleliseError(f"No new work folder could be made under {parent}. Try again.")


def _positive(kind: type) -> Callable[[str], float]:
    """An argparse type for a number that must be above zero."""

    def convert(text: str) -> float:
        value = kind(text)
        if value <= 0:
            raise argparse.ArgumentTypeError(f"{text} is not above zero")
        return value

    return convert


def _safe(text: str) -> str:
    """Make a model name safe to use as a folder name."""
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text) or "model"


# --- the commands ----------------------------------------------------------------------------


def cmd_serial(args: argparse.Namespace) -> int:
    """Have a model, at temperature zero, translate a notebook into the problem's serial routine.

    A serial routine that exists already is never replaced without ``--force``, because it is
    the referee and the baseline of every run.
    """
    folder, nb = _problem(args.problem)
    dest = folder / "serial.c"
    if dest.exists() and not args.force:
        raise ParalleliseError(f"{dest} exists already, and every judgement of {folder.name} "
                               "rests on it. Pass --force to replace it.")  # fmt: skip
    _require_tools("serial")
    settings = _settings(args)
    settings.temperature = 0.0
    verdict, used = serial.translate(nb, settings, dest, step_budget=args.step_budget)
    print("\n".join(verdict.log))
    print(f"The model made {used['requests']} requests, with {used['prompt_tokens']} prompt and "
          f"{used['completion_tokens']} completion tokens.")  # fmt: skip
    if not verdict.correct:
        print(f"The translation was not accepted, so {dest} was not written.")
        return 1
    print(f"The translation was accepted and written to {dest}. Read it before relying on it.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Have an agent write a problem's parallel routine, judge it, and record the run.

    The judgement takes whatever is on disk when the agent stops, whether it finished, used its
    steps, ran out of time or failed. The exit status is 0 when the routine is correct.
    """
    settings = _settings(args)
    folder, nb = _problem(args.problem)
    serial_c = folder / "serial.c"
    if not serial_c.is_file():
        raise ParalleliseError(f"The problem {folder.name} has no serial routine yet. Make it once "
                               f"with 'parallelise serial {folder}', read it, and run again.")
    decl = notebook.declaration(nb)
    store = ResultsStore(settings.results_path)
    store.ensure_writable()
    _require_tools(args.backend)
    agent = get_agent(args.agent)
    work_dir = _new_work_folder(Path(settings.work_root) / folder.name / args.backend
                                / _safe(settings.model) / args.agent)  # fmt: skip
    shown = convert.prepare(nb, decl, serial_c, args.backend, work_dir, alias=args.masked)
    name = shown.name
    started = time.perf_counter()
    with ModelClient(settings) as client:
        try:
            result = agent.run(work_dir, convert.task(name, args.backend), settings, client,
                               args.step_budget,
                               convert.toolset(nb, shown, work_dir, name, args.backend),
                               convert.prompt(name, args.backend))  # fmt: skip
        except AgentUnavailable:
            raise
        except Exception as exc:  # an agent that crashes is a result to record
            log.error("the agent failed: %s: %s", type(exc).__name__, exc,
                      exc_info=not isinstance(exc, ParalleliseError))  # fmt: skip
            result = AgentResult(False, error=f"{type(exc).__name__}: {exc}")
        used = client.meter.summary()
    agent_time = time.perf_counter() - started

    routine = work_dir / generate.routine_file(name, args.backend)
    verdict = judge.judge(nb, decl, routine, work_dir / "serial.c", args.backend,
                          timeout_s=args.check_timeout, name=name)  # fmt: skip
    (work_dir / "check.log").write_text("\n".join(verdict.log) + "\n", encoding="utf-8")
    notes = [args.notes, result.notes]
    if result.error:
        notes.append(f"The agent stopped with {result.error.rstrip('.')}.")
    if result.time_limit_reached:
        notes.append(f"The time limit of {settings.time_limit_s:g} seconds stopped the agent.")
    store.append(RunRecord(
        problem=folder.name, backend=args.backend, model=settings.model, agent=args.agent,
        correct=verdict.correct, failure_stage=verdict.stage, speedup=verdict.speedup,
        faster=verdict.faster, scaling=verdict.scaling, parallel=verdict.parallel,
        masked=bool(args.masked), repeat=args.repeat, serving=args.serving,
        quantisation=args.quantisation, temperature=settings.temperature,
        step_budget=args.step_budget, agent_time_s=round(agent_time, 2),
        cores=provenance.cores(), source_hash=provenance.source_hash(),
        serial_hash=hashlib.sha256(serial_c.read_bytes()).hexdigest()[:12],
        work_folder=str(work_dir), notes=" ".join(n for n in notes if n), **used,
    ))  # fmt: skip
    print(f"Judged {folder.name} on {args.backend} with {args.agent} and {settings.model}; "
          f"the record is in {settings.results_path} and the work in {work_dir}.")
    print(verdict_line(verdict))
    return 0 if verdict.correct else 1


def cmd_check(args: argparse.Namespace) -> int:
    """Judge the routine in a finished work folder again, with the problem it was made for.

    The exit status is 0 when the routine is correct, whatever its speed.
    """
    work_dir = Path(args.work_dir)
    _, nb = _problem(args.problem)
    _require_tools(args.backend)
    headers = sorted(work_dir.glob("*.h"))
    if len(headers) != 1 or not (work_dir / "serial.c").is_file():
        raise ParalleliseError(f"{work_dir} is not a work folder made by 'parallelise run': it "
                               "should hold one header and serial.c.")  # fmt: skip
    name = headers[0].stem
    routine = work_dir / generate.routine_file(name, args.backend)
    verdict = judge.judge(nb, notebook.declaration(nb), routine, work_dir / "serial.c",
                          args.backend, timeout_s=args.timeout, name=name)  # fmt: skip
    print("\n".join(verdict.log))
    print(verdict_line(verdict))
    return 0 if verdict.correct else 1


_COMPARED = {"agrees": "agrees with the C", "differs": "differs from the C",
             "not compared": "could not be compared with the C"}  # fmt: skip


def cmd_reverse(args: argparse.Namespace) -> int:
    """Have a model translate a parallel routine back into plain Python, and read the
    translation by the fixed rule. The exit status is 0 when the rule flags nothing."""
    folder, nb = _problem(args.problem)
    routine = Path(args.routine)
    if not routine.is_file():
        raise ParalleliseError(f"There is no routine at {routine}. Give its source file.")
    _require_tools(args.backend)
    settings = _settings(args)
    settings.temperature = 0.0
    out = Path(args.out) if args.out else _new_work_folder(
        Path(settings.work_root) / "reverse" / folder.name / args.backend)  # fmt: skip
    reading = reverse.translate_back(nb, notebook.declaration(nb), routine, args.backend,
                                     settings, out, name=args.name,
                                     step_budget=args.step_budget)  # fmt: skip
    print(f"The translation {_COMPARED[reading.compared]}.")
    print("The rule flags nothing." if not reading.flags else
          "The rule flags this:\n" + "\n".join(f"  - {flag}" for flag in reading.flags))
    if reading.sizes:
        print(f"The routine was checked again at the sizes {reading.sizes}.")
    if reading.opinion:
        print(f"The fixed model's opinion, which is no verdict: {reading.opinion}")
    used = reading.used
    print(f"The model made {used.get('requests', 0)} requests, with "
          f"{used.get('prompt_tokens', 0)} prompt and {used.get('completion_tokens', 0)} "
          "completion tokens.")  # fmt: skip
    print(f"The translation and the reading are in {out}.")
    return 1 if reading.flags else 0


def cmd_report(args: argparse.Namespace) -> int:
    """Print the results tables, and draw the chart of correct runs if asked."""
    path = _settings(args).results_path
    records = ResultsStore(path).load()
    if not records:
        raise ParalleliseError(f"There are no run records in {path}. Run 'parallelise run' "
                               "first, or give --results the path of a results file.")
    print(report.build_tables(records, report.load_prices(args.prices) if args.prices else None))
    if args.chart:
        report.build_chart(records, Path(args.chart))
        print(f"The chart is in {args.chart}.")
    return 0


# --- the parser ------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all its commands."""
    parser = argparse.ArgumentParser(
        prog="parallelise", description="Turn a serial Python notebook into a parallel C routine "
                                        "with a language-model agent, and judge the result.")
    parser.add_argument("--version", action="version", version=f"parallelise {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default="WARNING", help="DEBUG, INFO, WARNING or ERROR")
    endpoint = argparse.ArgumentParser(add_help=False)
    endpoint.add_argument("--gateway-url", help="the model endpoint (default: http://localhost:4000/v1)")
    endpoint.add_argument("--model", help="the model's name (default: local)")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    def command(name: str, parents: list, words: str, example: str,
                func: object) -> argparse.ArgumentParser:
        p = sub.add_parser(name, parents=parents, help=words,
                           description=words[0].upper() + words[1:] + ".",
                           epilog=f"Example: {example}")  # fmt: skip
        p.set_defaults(func=func)
        return p

    p = command("serial", [common, endpoint], "have a model, at temperature zero, translate a "
                "problem's notebook into its serial routine, serial.c, which names that model and "
                "is accepted only if it agrees with the Python at every test size and at the "
                "benchmark size", "parallelise serial problems/integrate --model kimi-k2.5",
                cmd_serial)
    p.add_argument("problem", help="a folder holding a notebook of the same name")
    p.add_argument("--step-budget", type=_positive(int), default=40,
                   help="the most requests the agent may make to the model (default: 40)")
    p.add_argument("--force", action="store_true", help="replace a serial routine that exists")

    p = command("run", [common, endpoint], "have an agent write a problem's parallel routine "
                "in a new work folder, judge it, and record the run",
                "parallelise run problems/integrate --backend openmp --model local", cmd_run)
    p.add_argument("problem", help="a folder holding the notebook and its serial.c")
    p.add_argument("--backend", required=True, choices=generate.BACKENDS)
    p.add_argument("--agent", choices=AGENT_NAMES, default="basic",
                   help="the design of agent that drives the model (default: basic)")
    p.add_argument("--masked", metavar="ALIAS",
                   help="hide the routine's name and every telling name behind this alias")
    p.add_argument("--step-budget", type=_positive(int), default=80,
                   help="the most requests the agent may make to the model (default: 80)")
    p.add_argument("--time-limit", type=_positive(float), metavar="SECONDS",
                   help="the agent's time limit (default: 1800)")
    p.add_argument("--check-timeout", type=_positive(float), default=900.0,
                   help="the seconds allowed for the judgement (default: 900)")
    p.add_argument("--repeat", type=_positive(int), default=1,
                   help="the number of this run, when it is repeated (default: 1)")
    p.add_argument("--work-root", help="the folder for work folders (default: work)")
    p.add_argument("--results", help="the results file (default: results/local-runs.jsonl)")
    p.add_argument("--serving", default="not recorded",
                   help="who served the model, such as its provider")
    p.add_argument("--quantisation", default="not disclosed",
                   help="the numeric precision of the model's weights, if known")
    p.add_argument("--notes", default="", help="free text kept with the record")

    p = command("check", [common], "judge the routine in a finished work folder again, without "
                "the agent or the model", "parallelise check work/integrate/openmp/local/basic/"
                "2026-11-07T09-30-00Z --problem problems/integrate --backend openmp", cmd_check)
    p.add_argument("work_dir", help="a work folder made by 'parallelise run'")
    p.add_argument("--problem", required=True, help="the problem folder it was made for")
    p.add_argument("--backend", required=True, choices=generate.BACKENDS)
    p.add_argument("--timeout", type=_positive(float), default=900.0,
                   help="the seconds allowed for the judgement (default: 900)")

    p = command("reverse", [common, endpoint], "have a model translate a parallel routine back "
                "into plain Python, and flag what a faithful numerical routine has no need for",
                "parallelise reverse work/integrate/openmp/local/basic/2026-11-08T09-30-00Z/"
                "integrate.c --problem problems/integrate --backend openmp --model kimi-k2.5",
                cmd_reverse)
    p.add_argument("routine", help="the routine's source file")
    p.add_argument("--problem", required=True, help="the problem folder it was written for")
    p.add_argument("--backend", required=True, choices=generate.BACKENDS)
    p.add_argument("--name", help="the routine's own name, if it is not the declared one, as "
                   "in a masked run")
    p.add_argument("--step-budget", type=_positive(int), default=40,
                   help="the most requests the agent may make to the model (default: 40)")
    p.add_argument("--out", help="the folder for the translation and the reading (default: a "
                   "new folder under work/reverse)")

    p = command("report", [common], "print the tables of correct runs, speed and model use, "
                "and draw a chart if asked", "parallelise report --results runs.jsonl "
                "--chart correct.svg", cmd_report)
    p.add_argument("--results", help="the results file (default: results/local-runs.jsonl)")
    p.add_argument("--prices", help="a JSON table of US dollars per million tokens, for costs")
    p.add_argument("--chart", help="write the chart of correct runs to this SVG file")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line and return the exit status."""
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:  # argparse exits itself; turn that into a return value
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 2)
    try:
        configure_logging(args.log_level)
        return int(args.func(args))
    except ParalleliseError as exc:
        print(f"parallelise: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:  # a path that cannot be read or written, not caught closer to it
        log.debug("operating-system error", exc_info=True)
        where = f": {exc.filename}" if exc.filename else ""
        print(f"parallelise: {exc.strerror or exc}{where}. Check the path and its permissions.",
              file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("parallelise: interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
