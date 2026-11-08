"""Step 1: an independent agent translates a notebook's Python function into plain C.

The agent knows nothing of the rest of the work. It is told only to translate the function
faithfully into C that gives the same results. It sees the function as the notebook defines it
and the header that gives the C prototype; it may read those files and write one, its
translation. Its check builds the translation with the serial driver and compares it with the
Python on fresh inputs at every test size. The run ends when the check passes, when the step
budget is spent, or when the time limit passes.

A translation is accepted only after one more comparison, on new inputs at every test size and
at the benchmark size. The author then reads it, and it becomes the problem's serial routine
(``serial.c`` beside the notebook): the referee of correctness at the benchmark size and the
baseline of every speed-up.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import generate, judge, notebook
from .agents.base import Toolset, file_toolset
from .agents.basic import BasicAgent
from .config import Settings
from .model import ModelClient
from .routine import Declaration

PROMPT = (
    "You translate a Python function into C. The file function.py holds the Python, and "
    "{header} gives the C prototype of the function to write. Write {source}: C11 that gives "
    "the same results as the Python function for every input of the declared shapes, using "
    "only the C standard library. Translate the function faithfully and plainly. Every "
    "function other than {name} must be static. Call the tool check to compare your C with the "
    "Python on fresh inputs. When check reports that they agree, reply that you have finished."
)


def toolset(nb: Path, decl: Declaration, folder: Path) -> Toolset:
    """The translator's tools: read the Python and the header, write its translation, check it."""
    source = f"{decl.name}.c"

    def report(verdict: judge.Verdict) -> str:
        return ("check passed: " if verdict.correct else "check failed: ") + "; ".join(verdict.log)

    def finish() -> tuple[bool, str]:
        verdict = judge.correct_at(nb, decl, folder / source, "serial")
        return verdict.correct, report(verdict) + "\nFix the C file, call check, then finish."

    return file_toolset(folder, ["function.py", f"{decl.name}.h"], source,
                        "Build your C file and compare it with the Python.",
                        lambda: report(judge.correct_at(nb, decl, folder / source, "serial",
                                                        list(decl.test_sizes))),
                        finish)  # fmt: skip


def accepted_sizes(serial_c: Path) -> list[int] | None:
    """The sizes at which a serial routine was accepted, read from the comment at its head."""
    head = serial_c.read_text(encoding="utf-8", errors="replace").split("\n", 1)[0]
    found = re.search(r"accepted on [\d-]+ at the sizes \[([\d, ]*)\]", head)
    return [int(s) for s in found.group(1).split(",") if s.strip()] if found else None


def note(nb: Path, model: str, sizes: list[int]) -> str:
    """The comment at the head of an accepted serial routine, which says where it came from."""
    return (f"/* The serial routine of {nb.name}, translated from its Python by the model {model} "
            f"at temperature 0 and accepted on {datetime.now(UTC).date().isoformat()} at the "
            f"sizes {sizes}. */\n")


def translate(nb: Path, settings: Settings, dest: Path,
              step_budget: int = 40) -> tuple[judge.Verdict, dict[str, Any]]:
    """Have the model in ``settings`` translate the notebook's function, and judge the result.

    The translation is written to ``dest`` only when it is accepted, beneath a comment that
    names the model, the date and the sizes of the acceptance. Returns the verdict and the
    model's use: its requests and tokens.
    """
    decl = notebook.declaration(nb)
    folder = Path(tempfile.mkdtemp(prefix="parallelise-serial-"))
    try:
        (folder / "function.py").write_text(notebook.python_source(nb), encoding="utf-8")
        (folder / f"{decl.name}.h").write_text(generate.header(decl, decl.name), encoding="utf-8")
        prompt = PROMPT.format(header=f"{decl.name}.h", source=f"{decl.name}.c", name=decl.name)
        task = f"Translate the function {decl.name} in function.py into {decl.name}.c."
        with ModelClient(settings) as client:
            BasicAgent().run(folder, task, settings, client, step_budget,
                             toolset(nb, decl, folder), prompt)  # fmt: skip
            used = client.meter.summary()
        written = folder / f"{decl.name}.c"
        if not written.is_file():
            return judge.Verdict(False, "no-files", log=["no translation was written"]), used
        sizes = [*decl.test_sizes, decl.benchmark_size]
        verdict = judge.correct_at(nb, decl, written, "serial", sizes, timeout_s=3600)
        if verdict.correct:
            dest.write_text(note(nb, settings.model, sizes) + written.read_text(encoding="utf-8"),
                            encoding="utf-8")  # fmt: skip
        return verdict, used
    finally:
        shutil.rmtree(folder, ignore_errors=True)
