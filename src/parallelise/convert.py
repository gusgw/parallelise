"""Step 2: the model under study writes the parallel routine.

The agent works in a folder that holds three files it may read: the notebook's Python
(``function.py``), the problem's serial routine (``serial.c``) and the generated header.
In a study run all three are masked: the routine's name becomes an alias, and comments and
telling names are removed. The agent writes one file, its routine. Its check builds the
routine, compares it with the Python on fresh inputs at the test sizes, and times it
against the serial routine at the benchmark size. It finishes when the check finds the
routine correct and at least ``AGENT_GOAL`` times as fast as the serial routine; a routine
that never gets there is still judged when the agent stops.

The agent never sees an expected value, a file it could alter other than its own routine, or
the code that judges it.
"""

from __future__ import annotations

from pathlib import Path

from . import generate, judge, notebook
from .agents.base import Toolset, file_toolset
from .routine import Declaration
from .study.mask import mask_sources

#: The speed-up over the serial routine at which an agent may finish.
AGENT_GOAL = 1.2

BACKEND_WORDS = {
    "openmp": "OpenMP threads on one computer",
    "mpi": "MPI processes; every process receives all the arguments, and process 0 must hold "
    "every result when the routine returns",
    "cuda": "an NVIDIA graphics card with CUDA; the arguments and results are in ordinary "
    "memory, so the routine copies them to the card and back",
}

PROMPT = (
    "You write a parallel version of a numerical routine in C, using {words}. The file "
    "function.py holds the routine in Python, serial.c holds a plain C translation of it, and "
    "{header} gives the prototype of the routine to write and the rules it must keep. Write "
    "{source}: a routine that gives the same results as the Python for every input of the "
    "declared shapes, and runs faster than serial.c. Every function other than {name} must be "
    "static. Call the tool check to build your routine, compare it with the Python on fresh "
    "inputs and time it against serial.c. When check reports that it is correct and faster, "
    "reply that you have finished."
)

def prepare(nb: Path, decl: Declaration, serial: Path, backend: str, folder: Path,
            alias: str | None = None) -> Declaration:
    """Write the files the agent may read into ``folder``, and return the declaration as the
    agent sees it, whose name is the routine's name there.

    In a masked run that declaration has the masked names, so that the agent's check builds
    against the header the agent read and names the arguments and results as the agent knows
    them.
    """
    python, c_source = notebook.python_source(nb), serial.read_text(encoding="utf-8")
    shown = decl
    if alias:
        python, c_source, names = mask_sources(python, c_source, decl, alias)
        shown = decl.renamed(names)
    (folder / "function.py").write_text(python, encoding="utf-8")
    (folder / "serial.c").write_text(c_source, encoding="utf-8")
    (folder / f"{shown.name}.h").write_text(generate.header(shown, shown.name), encoding="utf-8")
    return shown


def prompt(name: str, backend: str) -> str:
    return PROMPT.format(words=BACKEND_WORDS[backend], header=f"{name}.h",
                         source=generate.routine_file(name, backend), name=name)  # fmt: skip


def task(name: str, backend: str) -> str:
    return f"Write {generate.routine_file(name, backend)}, the parallel version of {name}."


def toolset(nb: Path, decl: Declaration, folder: Path, name: str, backend: str,
            cache: Path) -> Toolset:
    """The agent's tools: read its three files and its routine, write the routine, check it.

    ``decl`` is the declaration that ``prepare`` returned, so that in a masked run the check
    shows only masked names.

    ``cache`` is a folder outside the agent's reach in which the check keeps the serial
    routine's answers and times at the benchmark size; the caller deletes it afterwards.
    """
    source = generate.routine_file(name, backend)

    def report(verdict: judge.Verdict) -> str:
        if verdict.stage == "notebook":  # its message names the notebook, which masking hides
            return "check failed: the Python could not be run, which is no fault of your routine"
        if not verdict.correct:
            return "check failed: " + "; ".join(verdict.log)
        faster = (verdict.speedup or 0) >= AGENT_GOAL
        return ("check passed: correct, and faster than serial.c; " if faster else
                "check: correct, but not yet faster than serial.c; ") + "; ".join(verdict.log)

    def check() -> judge.Verdict:
        return judge.quick_check(nb, decl, folder / source, folder / "serial.c", backend, name,
                                 cache)  # fmt: skip

    def finish() -> tuple[bool, str]:
        verdict = check()
        done = verdict.correct and (verdict.speedup or 0) >= AGENT_GOAL
        return done, report(verdict) + "\nImprove the routine, call check, then finish."

    return file_toolset(folder, ["function.py", "serial.c", f"{name}.h"], source,
                        "Build your routine, compare it with the Python and time it against "
                        "serial.c.", lambda: report(check()), finish)
