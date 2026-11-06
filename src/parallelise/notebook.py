"""Run a notebook's code, and ask its functions for inputs and for the trusted answers.

The notebook's code cells run in a child Python process with a time limit, so that a
notebook that fails, loops or exits cannot take the harness down with it. Each request runs
the cells once and then does all its work: reading the declaration, building inputs from
seeds, or computing the answers for input files. Inputs and answers pass through the binary
files described in ``routine``.

Run as a program (``python -m parallelise.notebook NOTEBOOK REQUEST REPLY``), this module is
the child's side of the exchange.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ParalleliseError
from .routine import (
    Declaration,
    DeclarationError,
    as_results,
    extents,
    parse_declaration,
    read_inputs,
    read_sizes,
    sizes_of,
    write_inputs,
    write_results,
)

DEFAULT_TIMEOUT_S = 600.0


class NotebookError(ParalleliseError):
    """A notebook that cannot be run, or whose functions do not match their declaration."""


def _ask(notebook: Path, request: dict[str, Any], timeout_s: float) -> Any:
    """Send one request to a child process that runs the notebook, and return its reply."""
    notebook = Path(notebook).resolve()
    if not notebook.is_file():
        raise NotebookError(f"The notebook {notebook} does not exist. Check the path.")
    with tempfile.TemporaryDirectory(prefix="parallelise-notebook-") as tmp:
        ask, reply = Path(tmp) / "request.json", Path(tmp) / "reply.json"
        ask.write_text(json.dumps(request), encoding="utf-8")
        command = [sys.executable, "-m", "parallelise.notebook", str(notebook), str(ask)]
        command.append(str(reply))
        try:
            proc = subprocess.run(
                command, cwd=notebook.parent, capture_output=True, text=True,
                timeout=timeout_s, stdin=subprocess.DEVNULL, check=False,
            )  # fmt: skip
        except subprocess.TimeoutExpired:
            raise NotebookError(
                f"{notebook.name} took longer than {timeout_s:g} seconds. Make its cells and "
                "its functions quicker, or give the judgement more time (--check-timeout)."
            ) from None
        if proc.returncode != 0:
            message = proc.stderr.strip().splitlines()[-1:] or [f"exit status {proc.returncode}"]
            raise NotebookError(f"In {notebook.name}, {message[0]}")
        return json.loads(reply.read_text(encoding="utf-8"))


def declaration(notebook: Path, timeout_s: float = DEFAULT_TIMEOUT_S) -> Declaration:
    """Run the notebook and return its checked declaration.

    The functions are called once at every test size, and their inputs and results are
    checked against the declared types and shapes.
    """
    found = _ask(notebook, {"task": "declare"}, timeout_s)
    return parse_declaration(found["routine"], found["name"])


def write_input_files(
    notebook: Path, calls: list[tuple[int, int, Path]], timeout_s: float = DEFAULT_TIMEOUT_S
) -> list[dict[str, int]]:
    """Build the inputs of each (size, seed, file) and write them; return each call's sizes."""
    request = {"task": "inputs", "calls": [[size, seed, str(path)] for size, seed, path in calls]}
    return list(_ask(notebook, request, timeout_s))


def write_answer_files(
    notebook: Path, pairs: list[tuple[Path, Path]], timeout_s: float = DEFAULT_TIMEOUT_S
) -> None:
    """Call the notebook's function on each input file and write its results beside it."""
    _ask(notebook, {"task": "answers", "pairs": [[str(i), str(o)] for i, o in pairs]}, timeout_s)


def python_source(notebook: Path) -> str:
    """Return the notebook's reference function, with the functions, names and imports it uses.

    The code is read from the cells up to the one that declares ROUTINE. The imports are kept,
    and so is the function that ROUTINE names, with every function and module-level name that
    it uses, directly or through another; nothing else is. Demonstrations are left out because
    they can give away what the function computes, and ``make_inputs`` and ROUTINE because they
    serve the harness and not the translation.
    """
    statements: list[tuple[ast.stmt, str]] = []
    for cell in json.loads(Path(notebook).read_text(encoding="utf-8")).get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        try:
            body = ast.parse(source).body
        except SyntaxError:
            continue
        lines = source.splitlines()
        for node in body:
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            statements.append((node, "\n".join(lines[first - 1 : node.end_lineno])))
        if any(_declares_routine(n) for n in body):
            break
    wanted = [_reference_name(n) for n, _ in statements if _declares_routine(n)]
    queue = [w for w in wanted if w] or [n.name for n, _ in statements
                                         if isinstance(n, ast.FunctionDef)]  # fmt: skip
    kept = {i for i, (node, _) in enumerate(statements)
            if isinstance(node, (ast.Import, ast.ImportFrom))}  # fmt: skip
    while queue:
        name = queue.pop()
        for i, (node, _) in enumerate(statements):
            if i not in kept and name in _bound(node) and not _serves_the_harness(node):
                kept.add(i)
                queue.extend(n.id for n in ast.walk(node) if isinstance(n, ast.Name))
    return "\n\n\n".join(text for i, (_, text) in enumerate(statements) if i in kept) + "\n"


def _bound(node: ast.stmt) -> set[str]:
    """The module-level names that a statement defines."""
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in node.names}
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    return set()


def _declares_routine(node: ast.stmt) -> bool:
    return isinstance(node, ast.Assign) and "ROUTINE" in _bound(node)


def _reference_name(node: ast.stmt) -> str | None:
    """The name of the function that a ROUTINE dictionary written out in full declares."""
    value = getattr(node, "value", None)
    if isinstance(value, ast.Dict):
        for key, item in zip(value.keys, value.values, strict=True):
            if isinstance(key, ast.Constant) and key.value == "function":
                return item.id if isinstance(item, ast.Name) else None
    return None


def _serves_the_harness(node: ast.stmt) -> bool:
    return _declares_routine(node) or (isinstance(node, ast.FunctionDef)
                                       and node.name == "make_inputs")  # fmt: skip


# --- the child's side --------------------------------------------------------------------------


def _run_cells(notebook: Path) -> dict[str, Any]:
    cells = json.loads(notebook.read_text(encoding="utf-8")).get("cells", [])
    namespace: dict[str, Any] = {"__name__": "__main__"}
    for number, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = cell.get("source", "")
            source = "".join(source) if isinstance(source, list) else source
            try:
                exec(compile(source, f"{notebook.name} cell {number}", "exec"), namespace)
            except Exception as exc:
                raise NotebookError(f"cell {number} fails: {type(exc).__name__}: {exc}") from exc
    if not isinstance(namespace.get("ROUTINE"), dict):
        raise NotebookError("no cell defines the ROUTINE dictionary that declares the routine.")
    return namespace["ROUTINE"]


def _check_results(decl: Declaration, sizes: dict[str, int], results: dict[str, Any]) -> None:
    for value in decl.results:
        data = np.asarray(results[value.name])
        shape = extents(value, sizes)
        if (data.dtype.kind in "iu") != value.dtype.startswith("int") and data.dtype.kind != "b":
            raise DeclarationError(
                f"the result {value.name!r} holds {data.dtype}, not {value.dtype}"
            )
        fits = data.ndim == 1 and data.size <= shape[0] if value.bounded else data.shape == shape
        if not fits:
            raise DeclarationError(
                f"the result {value.name!r} has shape {data.shape}, but the declaration makes it "
                f"{'at most ' if value.bounded else ''}{shape}"
            )
        if data.dtype.kind == "f" and not np.all(np.isfinite(data)):
            raise DeclarationError(f"the result {value.name!r} holds a value that is not finite")


def _serve(notebook: Path, request: dict[str, Any]) -> Any:
    routine = _run_cells(notebook)
    function, make_inputs = routine.get("function"), routine.get("make_inputs")
    if not callable(function) or not callable(make_inputs):
        raise NotebookError("ROUTINE must name the reference function and make_inputs.")
    static = {k: v for k, v in routine.items() if k not in ("function", "make_inputs")}
    decl = parse_declaration(static, function.__name__)
    task = request["task"]
    if task == "declare":
        names = sorted(a.name for a in decl.arguments)
        for seed, size in enumerate(decl.test_sizes):
            inputs = make_inputs(size, np.random.default_rng(seed))
            if sorted(inputs) != names:
                raise DeclarationError(f"make_inputs returns {sorted(inputs)}, not {names}")
            _check_results(decl, sizes_of(decl, inputs), as_results(decl, function(**inputs)))
        return {"routine": static, "name": function.__name__}
    if task == "inputs":
        return [
            write_inputs(Path(path), decl, make_inputs(int(size), np.random.default_rng(seed)))
            for size, seed, path in request["calls"]
        ]
    if task == "answers":
        for inputs_path, results_path in request["pairs"]:
            results = as_results(decl, function(**read_inputs(Path(inputs_path), decl)))
            _check_results(decl, read_sizes(Path(inputs_path), decl), results)
            write_results(Path(results_path), decl, results)
        return None
    raise NotebookError(f"unknown request {task!r}")


if __name__ == "__main__":
    notebook_path, request_path, reply_path = (Path(a) for a in sys.argv[1:4])
    try:
        reply = _serve(notebook_path, json.loads(request_path.read_text(encoding="utf-8")))
    except (ParalleliseError, KeyError, TypeError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        sys.exit(1)
    reply_path.write_text(json.dumps(reply), encoding="utf-8")
