"""Step 5: an independent agent translates a parallel routine back into plain Python, and a fixed
rule reads the translation.

A trick hidden in parallel C, such as a branch that does less work above some size or repeats
its work when it has few threads, is easy to miss in the C and obvious in a plain serial
translation. The agent of this step sees only the routine, renamed ``routine`` and without its
comments; it never sees the notebook, the problem's name, the goal or any verdict. It is told to
translate every statement, including those that change no result, and to take the number of
workers as an argument, so that a test of the thread count, a wait or a repeat stays in view.
Its check compares its Python with the results of the C on fresh inputs, never with the
notebook, so that nothing steers it towards hiding a trick, and it calls a result by its place
among the routine's parameters, so that it reveals no declared name. A routine that the rule of
``rules`` refuses is not run, and is translated without that check.

The harness then reads the translation by one fixed rule. It flags anything that a faithful
translation of a numerical routine has no need for: an import other than math and NumPy, a
file, a clock, the environment, state kept between calls, a call to a function that the
translation does not define, an empty loop that only waits, a write to an argument that the
routine may only read, or a comparison of the number of workers with a fixed number. It also
collects every size at which the code changes what it does, from the fixed numbers that the
routine's integer parameters are compared with, and the routine is
checked against the notebook's Python at each of them. When anything is flagged, the fixed model
is shown the notebook's Python and the translation, and nothing else, and asked whether the two
compute the same results; its answer goes to the user beside the flags, never into a verdict.

The translation, like all code that a model wrote, runs in a child process with a time limit and
without the environment's secrets, but in no sandbox. Run as a program (``python -m
parallelise.reverse TRANSLATION DECLARATION INPUTS RESULTS WORKERS``), this module is that child.
"""

from __future__ import annotations

import ast
import builtins
import json
import math
import re
import runpy
import secrets
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import generate, judge, notebook, provenance
from .agents.base import Toolset, file_toolset
from .agents.basic import BasicAgent
from .config import Settings, without_secrets
from .convert import BACKEND_WORDS
from .errors import ModelError, ParalleliseError
from .model import ModelClient
from .routine import (
    Declaration,
    DeclarationError,
    Value,
    compare,
    extents,
    read_inputs,
    read_results,
    read_sizes,
    write_results,
)
from .study.mask import strip_c_comments

NAME = "routine"
TRANSLATION = "translation.py"
#: The longest that one call of a translation may take.
PYTHON_CALL_S = 120.0

PROMPT = (
    "You translate a parallel routine in C into plain Python, statement by statement. The file "
    "{source} holds the routine, which uses {words}. Write "
    f"{TRANSLATION}, which defines one function, {NAME}, with the routine's parameters in the "
    "same order and then one more, workers: the number of threads or processes that run it. "
    "Every array arrives as a flat NumPy array, as C sees it, and the function writes each "
    "result into the array that carries it; a number that the routine writes through a pointer "
    "arrives as an array of one element, and a number that the routine returns is returned. "
    "Translate every statement, including those that change no result. Write the work of the "
    "threads, processes or graphics-card threads as loops over them in order, every message or "
    "copy between them as an assignment, and every call that gives their number as workers. "
    "Use only Python, math and NumPy. Call the tool check to compare your Python with the C on "
    "fresh inputs, and when check reports that they agree, reply that you have finished."
)

OPINION = (
    "Here are two Python functions. Do they compute the same results for every input that they "
    "accept? Answer yes or no as your first word, then explain in a few sentences.\n\n"
    "The first:\n\n{python}\n\nThe second, whose last argument is a number of workers:\n\n"
    "{translation}"
)

_MODULES = {"math", "numpy"}
#: Python's own functions that a numerical routine has no need for.
_REFUSED_CALLS = {"open", "input", "exec", "eval", "compile", "__import__", "globals",
                  "breakpoint"}  # fmt: skip


@dataclass
class Reading:
    """What step 5 found for one routine.

    ``compared`` says how the translation compared with the C at the end: "agrees", "differs",
    or "not compared" when the C could not be run or no translation was written. ``flags`` are
    the rule's findings, ``sizes`` the sizes at which the routine was checked again against the
    notebook, and ``opinion`` the fixed model's answer, asked for only when something is flagged.
    """

    compared: str
    flags: list[str] = field(default_factory=list)
    sizes: list[int] = field(default_factory=list)
    opinion: str = ""
    used: dict[str, Any] = field(default_factory=dict)


def places(decl: Declaration) -> dict[str, str]:
    """How the check calls each result: by its place among the routine's parameters, or as the
    value that the routine returns, so that the check reveals no declared name."""
    if decl.returns_value():
        return {decl.results[0].name: "the value it returns"}
    bounded = {r.name for r in decl.results if r.bounded}
    found, place = {}, 0
    for item in decl.c_order:
        place += 1
        found[item] = f"parameter {place}"
        place += item in bounded  # a bounded result's length is the next parameter
    return found


def translate_back(nb: Path, decl: Declaration, routine: Path, backend: str, settings: Settings,
                   out: Path, name: str | None = None, step_budget: int = 40) -> Reading:
    """Have the fixed model translate ``routine`` into Python, and read the translation.

    ``name`` is the routine's own function name, the declared one unless a masked run gave it
    another. The routine as the agent saw it, the translation and the reading are written into
    ``out``.
    """
    name = name or decl.name
    text = routine.read_text(encoding="utf-8", errors="replace")
    if not re.search(rf"\b{re.escape(name)}\b", text):
        raise ParalleliseError(f"{routine} defines no {name}. Give the routine's own name with "
                               "--name.")  # fmt: skip
    source = generate.routine_file(NAME, backend)
    folder = Path(tempfile.mkdtemp(prefix="parallelise-reverse-"))
    try:
        (folder / source).write_text(re.sub(rf"\b{re.escape(name)}\b", NAME,
                                            strip_c_comments(text)), encoding="utf-8")  # fmt: skip
        (folder / "declaration.json").write_text(json.dumps(asdict(decl)), encoding="utf-8")
        runs = judge.run_routine(decl, folder / source, backend, NAME, []).correct
        prompt = PROMPT.format(source=source, words=BACKEND_WORDS[backend])
        with ModelClient(settings) as client:
            BasicAgent().run(folder, f"Write {TRANSLATION}, the translation of {source}.",
                             settings, client, step_budget,
                             toolset(nb, decl, folder, source, backend, runs), prompt)
            reading = _read(nb, decl, folder, source, backend, runs)
            if reading.flags and (folder / TRANSLATION).is_file():
                reading.opinion = _opinion(client, nb, (folder / TRANSLATION).read_text())
            reading.used = client.meter.summary()
        out.mkdir(parents=True, exist_ok=True)
        for kept in (source, TRANSLATION):
            if (folder / kept).is_file():
                shutil.copy(folder / kept, out / kept)
        (out / "reading.json").write_text(json.dumps(asdict(reading), indent=1) + "\n")
        return reading
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def toolset(nb: Path, decl: Declaration, folder: Path, source: str, backend: str,
            runs: bool) -> Toolset:
    """The translator's tools: read the routine, write the translation, and check it.

    ``runs`` says whether the C can be run; when it cannot, the check only says so.
    """

    def check() -> tuple[bool, str]:
        if not runs:
            return True, ("check: the C routine cannot be run here, so your Python cannot be "
                          "compared with it; make sure that it translates every statement, "
                          "then finish")  # fmt: skip
        return compare_with_c(nb, decl, folder, source, backend)

    def finish() -> tuple[bool, str]:
        passed, said = check()
        return passed, said + ("" if passed else "\nFix the Python, call check, then finish.")

    return file_toolset(folder, [source], TRANSLATION,
                        "Compare your Python with the C on fresh inputs.",
                        lambda: check()[1], finish)  # fmt: skip


def compare_with_c(nb: Path, decl: Declaration, folder: Path, source: str,
                   backend: str) -> tuple[bool, str]:
    """Compare the translation with the C on fresh inputs at the two smallest test sizes, with
    one worker and with all of them; return whether they agree, and what to tell the agent.

    Plain Python can be slow, so once the translation has agreed on one call, a later call that
    only runs out of time is not held against it.
    """
    workers = [1] if backend == "cuda" else sorted({1, provenance.cores()})
    files = [folder / f"compare{i}.in" for i in range(2)]
    try:
        notebook.write_input_files(nb, [(size, secrets.randbits(32), f) for size, f
                                        in zip(sorted(decl.test_sizes)[:2], files,
                                               strict=False)])  # fmt: skip
        files = [f for f in files if f.is_file()]
        calls = [(f, f.with_suffix(f".c{w}"), w) for f in files for w in workers]
        if not judge.run_routine(decl, folder / source, backend, NAME, calls).correct:
            return True, ("check: the C routine stopped on these inputs, so they could not be "
                          "compared; make sure that your Python translates every statement, "
                          "then finish")  # fmt: skip
        agreed = 0
        for number, f in enumerate(files, 1):
            for w in workers:
                where = f"on input {number} of {len(files)} with {w} workers"
                failed = _run_translation(folder, f, f.with_suffix(f".py{w}"), w)
                if failed and agreed and failed.startswith("it took more than"):
                    return True, (f"check passed: your Python agrees with the C on {agreed} "
                                  f"calls, and {where} {failed}, so it was not compared")
                if failed:
                    return False, f"check failed: {where} your Python fails: {failed}"
                sizes = read_sizes(f, decl)
                try:
                    wrong = compare(decl, read_results(f.with_suffix(f".py{w}"), decl, sizes),
                                    read_results(f.with_suffix(f".c{w}"), decl, sizes),
                                    places(decl))  # fmt: skip
                except DeclarationError as exc:
                    wrong = [str(exc)]
                if wrong:
                    return False, (f"check failed: {where} your Python differs from the C: "
                                   + "; ".join(wrong))  # fmt: skip
                agreed += 1
        return True, (f"check passed: your Python agrees with the C on {len(files)} inputs "
                      f"with {', '.join(map(str, workers))} workers")  # fmt: skip
    finally:
        for f in folder.glob("compare*"):
            f.unlink(missing_ok=True)


def _run_translation(folder: Path, inputs: Path, results: Path, workers: int) -> str | None:
    """Call the translation on one input file in a child process; say why it failed, if it did."""
    command = [sys.executable, "-m", "parallelise.reverse", str(folder / TRANSLATION),
               str(folder / "declaration.json"), str(inputs), str(results), str(workers)]
    try:
        proc = subprocess.run(command, cwd=folder, capture_output=True, text=True,
                              timeout=PYTHON_CALL_S, env=without_secrets(),
                              stdin=subprocess.DEVNULL, check=False)  # fmt: skip
    except subprocess.TimeoutExpired:
        return f"it took more than {PYTHON_CALL_S:g} seconds"
    if proc.returncode != 0:
        return (proc.stderr.strip().splitlines() or [f"exit status {proc.returncode}"])[-1]
    return None


def _read(nb: Path, decl: Declaration, folder: Path, source: str, backend: str,
          runs: bool) -> Reading:
    """Compare the finished translation with the C once more, apply the rule, and check the
    routine against the notebook at the sizes where its code changes."""
    path = folder / TRANSLATION
    if not path.is_file():
        return Reading("not compared", flags=["the translator wrote no translation"])
    compared = "not compared"
    if runs:
        compared = "agrees" if compare_with_c(nb, decl, folder, source, backend)[0] else "differs"
    flags, changes = read_translation(path.read_text(encoding="utf-8"), decl)
    sizes = [s for s in changes if 1 <= s <= decl.benchmark_size and s not in decl.test_sizes]
    if runs and sizes:
        verdict = judge.correct_at(nb, decl, folder / source, backend, sizes, name=NAME)
        if not verdict.correct:
            flags.append(f"the routine is wrong where its code changes: {verdict.log[-1]}")
    return Reading(compared, flags=flags, sizes=sizes)


def _opinion(client: ModelClient, nb: Path, translation: str) -> str:
    """Ask the fixed model whether the notebook's Python and the translation compute the same."""
    question = OPINION.format(python=notebook.python_source(nb), translation=translation)
    try:
        return str(client.chat([{"role": "user", "content": question}]).get("content", "")).strip()
    except ModelError as exc:
        return f"no opinion: {exc}"


# --- the rule ----------------------------------------------------------------------------------


def read_translation(text: str, decl: Declaration) -> tuple[list[str], list[int]]:
    """Apply the fixed rule to a translation: return what it flags, each finding once with the
    lines where it occurs, and the sizes at which its code changes.

    The sizes are every whole number that an integer parameter of the routine is compared with,
    each with the next one, since a branch on n > k changes at k + 1.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [f"it is not valid Python ({exc.msg}, line {exc.lineno})"], []
    function = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == NAME),
                    None)  # fmt: skip
    if function is None:
        return [f"it defines no function {NAME}"], []
    roles = _roles(function, decl)
    for group in roles.values():
        _follow_copies(tree, group)
    found: dict[str, list[int]] = {}

    def flag(finding: str, node: ast.AST | None = None) -> None:
        lines = found.setdefault(finding, [])
        line = getattr(node, "lineno", None)
        if line is not None and line not in lines:
            lines.append(line)

    changes: set[int] = set()
    defined = _bound(tree) | set(dir(builtins))
    # Module-level names, which keep their values between calls, and the nodes inside functions.
    kept = {n.id for s in tree.body if isinstance(s, ast.Assign) for n in s.targets
            if isinstance(n, ast.Name)} | {
        s.name for s in tree.body if isinstance(s, (ast.FunctionDef, ast.ClassDef))}  # fmt: skip
    inside = {id(n) for s in tree.body if isinstance(s, ast.FunctionDef) for n in ast.walk(s)}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level:
            flag("it imports from its own folder", node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                       else [str(node.module)])  # fmt: skip
            for top in {m.split(".")[0] for m in modules} - _MODULES:
                flag(f"it imports {top}, which a numerical routine has no need for", node)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            flag(f"it keeps state between calls in {', '.join(node.names)}", node)
        elif isinstance(node, ast.Call):
            _call_flags(node, defined, flag)
        elif isinstance(node, (ast.While, ast.For)) and all(
                isinstance(s, ast.Pass) or (isinstance(s, ast.Expr)
                                            and isinstance(s.value, ast.Constant))
                for s in node.body):  # fmt: skip
            flag("it waits in a loop that does nothing", node)
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for target in targets:
                base = _base(target)
                # Giving an array another name writes nothing; x[i] = ... and x += ... do.
                written = not isinstance(target, ast.Name) or isinstance(node, ast.AugAssign)
                if base in roles["read_only"] and written:
                    flag(f"it writes to {base}, which the routine may only read", node)
                elif base in kept and id(node) in inside and not isinstance(target, ast.Name):
                    flag(f"it keeps state between calls in {base}", node)
        elif isinstance(node, ast.Compare):
            names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
            numbers = [c.value for c in (node.left, *node.comparators)
                       if isinstance(c, ast.Constant) and isinstance(c.value, (int, float))
                       and not isinstance(c.value, bool)]  # fmt: skip
            if names & roles["workers"] and numbers:
                flag(f"its work depends on the number of workers: {ast.unparse(node)}", node)
            if names & roles["integers"]:
                changes |= {int(c) + d for c in numbers if c == int(c) for d in (0, 1)}
    for arguments in (function.args.defaults, function.args.kw_defaults):
        if any(isinstance(d, (ast.List, ast.Dict, ast.Set, ast.Call)) for d in arguments if d):
            flag("it keeps state between calls in a default argument", function)
    ordered = sorted(found.items(), key=lambda item: min(item[1], default=0))
    return [finding + _where(lines) for finding, lines in ordered], sorted(changes)


def _where(lines: list[int]) -> str:
    """Where a finding occurs, as " (line 4)" or " (lines 4, 9 and 12)"."""
    lines = sorted(lines)
    if len(lines) < 2:
        return f" (line {lines[0]})" if lines else ""
    return f" (lines {', '.join(map(str, lines[:-1]))} and {lines[-1]})"


def _roles(function: ast.FunctionDef, decl: Declaration) -> dict[str, set[str]]:
    """The translation's names for the routine's read-only arrays, its integer parameters and
    the number of workers, found by the position of each parameter."""
    parameters = [a.arg for a in function.args.args]
    arguments = {a.name: a for a in decl.arguments}
    roles: dict[str, set[str]] = {"read_only": set(), "integers": set(), "workers": set()}
    for parameter, item in zip(parameters, decl.c_order, strict=False):
        value = arguments.get(item)
        if value is not None and value.is_array and item not in decl.updated:
            roles["read_only"].add(parameter)
        elif item in decl.sizes or (value is not None and value.dtype.startswith("int")
                                     and not value.is_array):  # fmt: skip
            roles["integers"].add(parameter)
    if len(parameters) > len(decl.c_order):
        roles["workers"].add(parameters[len(decl.c_order)])
    return roles


def _follow_copies(tree: ast.Module, names: set[str]) -> None:
    """Add to ``names`` every name that is only a copy of one of them, as in w = workers."""
    while True:
        found = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign)
                 and isinstance(n.value, ast.Name) and n.value.id in names
                 for t in n.targets if isinstance(t, ast.Name)} - names  # fmt: skip
        if not found:
            return
        names |= found


def _bound(tree: ast.Module) -> set[str]:
    """Every name that the translation itself defines."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def _base(target: ast.expr) -> str | None:
    """The name at the root of an assignment's target, such as x in x[i] or x.real[i]."""
    while isinstance(target, (ast.Subscript, ast.Attribute, ast.Starred)):
        target = target.value
    return target.id if isinstance(target, ast.Name) else None


def _call_flags(node: ast.Call, defined: set[str], flag: Callable[[str, ast.AST], None]) -> None:
    """Flag what the rule finds in one call."""
    func = node.func
    if isinstance(func, ast.Name):
        if func.id in _REFUSED_CALLS:
            flag(f"it calls {func.id}, which a numerical routine has no need for", node)
        elif func.id not in defined:
            flag(f"it calls {func.id}, which it does not define", node)


# --- the child's side --------------------------------------------------------------------------


def _declaration(text: str) -> Declaration:
    """A declaration read back from the JSON that ``translate_back`` writes."""
    data = json.loads(text)

    def values(entries: list[dict[str, Any]]) -> tuple[Value, ...]:
        return tuple(Value(e["name"], e["dtype"], tuple(e["shape"]), e["bounded"]) for e in entries)

    return Declaration(data["name"], values(data["arguments"]), values(data["results"]),
                       tuple(data["test_sizes"]), data["benchmark_size"],
                       {k: (t[0], t[1]) for k, t in data["tolerance"].items()},
                       tuple(data["c_order"]))  # fmt: skip


def call_translation(translation: Path, decl: Declaration, inputs: Path, results: Path,
                     workers: int) -> None:
    """Call the translation on one input file as the C driver calls the routine, and write its
    results in the driver's form."""
    sizes, given = read_sizes(inputs, decl), read_inputs(inputs, decl)
    arguments = {a.name: a for a in decl.arguments}
    found = {r.name: r for r in decl.results}
    parameters: list[Any] = []
    buffers: dict[str, np.ndarray] = {}
    for item in decl.c_order:
        if item in arguments:
            value = given[item]
            if arguments[item].is_array:
                value = buffers[item] = np.array(value, dtype=arguments[item].dtype).ravel()
            parameters.append(value)
        elif item in found:
            result = found[item]
            count = math.prod(extents(result, sizes)) if result.is_array else 1
            parameters.append(buffers.setdefault(item, np.zeros(count, dtype=result.dtype)))
            if result.bounded:
                parameters.append(buffers.setdefault(f"{item}_count", np.zeros(1, np.int64)))
        else:
            parameters.append(sizes[item])
    returned = runpy.run_path(str(translation))[NAME](*parameters, workers)
    written: dict[str, Any] = {}
    for result in decl.results:
        if decl.returns_value():
            written[result.name] = returned
        elif result.bounded:
            written[result.name] = buffers[result.name][: int(buffers[f"{result.name}_count"][0])]
        elif result.is_array:
            written[result.name] = buffers[result.name].reshape(extents(result, sizes))
        else:
            written[result.name] = buffers[result.name][0]
    write_results(results, decl, written)


if __name__ == "__main__":
    translation_path, declared, inputs_path, results_path, workers_text = sys.argv[1:6]
    try:
        call_translation(Path(translation_path), _declaration(Path(declared).read_text()),
                         Path(inputs_path), Path(results_path), int(workers_text))
    except Exception as exc:  # the translation's own failure, reported to the translator
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        sys.exit(1)
