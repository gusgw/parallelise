"""Masking a problem so that a model cannot recognise the algorithm from its names.

A language model may have seen the well-known algorithms many times, so a result on a
problem called "jacobi" may reflect memory rather than ability. ``mask_problem`` writes a
copy of a problem in which everything that names the algorithm is replaced by a neutral
alias, while every number, every line of logic and the oracle values stay as they were.
A masked problem must still pass its reference check; the tests check that.

What changes:

- the kernel name becomes the alias in file names, header guards, function names, the
  Makefile ``KERNEL`` variable, the benchmark and the reference solution;
- notebook markdown is replaced by one neutral sentence, and cell outputs are cleared;
- comments (and Python docstrings) are removed from code cells and from C and Makefile
  sources;
- identifiers (or parts of identifiers, split at underscores and capitals) that appear in
  the word list below are renamed to neutral names such as ``mk12``.

Library names are left alone: attributes such as ``math.pi``, imported names, built-ins,
``M_PI``, ``MPI_Scan``, ``omp_*``, CUDA names, names after ``::``, and the directive
words of OpenMP pragmas.
The oracle macros that start with ``REF_`` are also left alone, since they are generated
from the notebook's dictionary keys.
"""

from __future__ import annotations

import ast
import builtins
import json
import logging
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("parallelise.study.mask")

#: Words that name an algorithm or its domain. Matching is on whole identifier parts and
#: ignores case. The position in this tuple fixes the neutral name, so the output is stable.
WORD_LIST: tuple[str, ...] = (
    "jacobi", "heat", "stencil", "midpoint", "integrate", "integral", "integrals",
    "integration", "pi", "matmul", "matrix", "matrices", "gemm", "histogram", "histograms",
    "bin", "bins", "scan", "prefix", "nbody", "body", "bodies", "gravity", "gravitational",
    "cg", "conjugate", "gradient", "decay", "reservoir", "laplace", "poisson", "relaxation",
    "quadrature", "trapezoid", "trapezoidal", "simpson", "riemann", "cumulative", "cumsum",
    "particle", "particles", "echo", "esn", "leaky", "sor", "gauss", "seidel",
)  # fmt: skip

NEUTRAL_SENTENCE = (
    "This notebook implements a numerical routine in serial Python; its final cell defines "
    "the oracle that a parallel implementation must reproduce."
)

_C_SUFFIXES = {".c", ".h", ".cu", ".cuh", ".cpp", ".hpp"}
_PROTECTED_PREFIXES = ("MPI_", "omp_", "OMP_", "cuda", "CUDA", "__", "REF_", "M_", "cub", "thrust")
_SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}
_SKIP_SUFFIXES = {".o", ".pyc", ".csv"}
_PARTS = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z0-9]+|_+|.")
_IDENT = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class Substitution:
    """One kind of change made in one file, with how many times it was made."""

    file: str
    original: str
    replacement: str
    count: int


@dataclass
class MaskReport:
    """Everything that ``mask_problem`` changed, so that the masking can be audited."""

    kernel: str
    alias: str
    substitutions: list[Substitution] = field(default_factory=list)
    renamed_files: list[tuple[str, str]] = field(default_factory=list)
    comments_removed: int = 0
    markdown_cells_replaced: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a plain-text description of all changes, one per line."""
        lines = [f"masked {self.kernel!r} as {self.alias!r}"]
        lines += [f"renamed file {old} to {new}" for old, new in self.renamed_files]
        lines += [
            f"{s.file}: {s.original} -> {s.replacement} ({s.count} times)"
            for s in self.substitutions
        ]
        lines.append(f"comments and docstrings removed: {self.comments_removed}")
        lines.append(f"notebook markdown cells replaced: {self.markdown_cells_replaced}")
        lines += self.notes
        return "\n".join(lines)


def _match_case(sample: str, replacement: str) -> str:
    if sample.isupper():
        return replacement.upper()
    if sample.islower():
        return replacement.lower()
    return replacement[:1].upper() + replacement[1:].lower()


class _Renamer:
    """Decides the neutral name of an identifier and records what it changed."""

    def __init__(self, kernel: str, alias: str, prefix: str) -> None:
        self.kernel = kernel
        self.alias = alias
        self.prefix = prefix
        self.kernel_re = re.compile(
            r"(?<![A-Za-z0-9])" + re.escape(kernel) + r"(?![A-Za-z0-9])", re.IGNORECASE
        )
        self.neutral = {w: f"{prefix}{i}" for i, w in enumerate(WORD_LIST)}
        self.counts: Counter[tuple[str, str, str]] = Counter()
        self.current_file = ""

    def _record(self, original: str, replacement: str) -> None:
        if original != replacement:
            self.counts[(self.current_file, original, replacement)] += 1

    def kernel_only(self, text: str) -> str:
        """Replace only the kernel name, wherever it stands as a word or word part."""
        return self.kernel_re.sub(
            lambda m: self._note(m.group(0), _match_case(m.group(0), self.alias)), text
        )

    def _note(self, original: str, replacement: str) -> str:
        self._record(original, replacement)
        return replacement

    def identifier(self, token: str, words: bool = True) -> str:
        """Return the neutral form of one identifier (or the identifier unchanged)."""
        if token.startswith(_PROTECTED_PREFIXES):
            return token
        out = self.kernel_re.sub(lambda m: _match_case(m.group(0), self.alias), token)
        if words:
            pieces = []
            for part in _PARTS.findall(out):
                key = part.lower()
                if key in self.neutral and key != self.alias.lower():
                    part = _match_case(part, self.neutral[key])
                pieces.append(part)
            out = "".join(pieces)
        self._record(token, out)
        return out

    def text(self, source: str, words: bool = True) -> str:
        """Rename every identifier in free text (a C file, a Makefile, a file name)."""

        def replace(match: re.Match[str]) -> str:
            before = source[max(0, match.start() - 2) : match.start()]
            if before == "::":
                return match.group(0)  # a name inside a library namespace
            return self.identifier(match.group(0), words)

        return _IDENT.sub(replace, source)

    def c_source(self, source: str) -> str:
        """Rename identifiers in C text.

        On OpenMP pragma lines only the text inside parentheses is renamed, because the
        directive words outside them (such as ``scan``) belong to OpenMP, whereas the
        variable names in clauses belong to the program.
        """
        out = []
        for line in source.splitlines(keepends=True):
            if line.lstrip().startswith("#pragma"):
                line = re.sub(r"\([^()]*\)", lambda m: self.text(m.group(0)), line)
            else:
                line = self.text(line)
            out.append(line)
        return "".join(out)


def strip_c_comments(source: str) -> tuple[str, int]:
    """Remove comments from C source, respecting strings and character literals.

    Returns the new text and the number of comments removed. Runs of blank lines left
    behind are collapsed to one.
    """
    out: list[str] = []
    removed = 0
    i, n = 0, len(source)
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if ch in "\"'":
            j = i + 1
            while j < n and source[j] != ch:
                j += 2 if source[j] == "\\" else 1
            out.append(source[i : j + 1])
            i = j + 1
        elif ch == "/" and nxt == "*":
            end = source.find("*/", i + 2)
            i = n if end < 0 else end + 2
            removed += 1
            out.append(" ")
        elif ch == "/" and nxt == "/":
            end = source.find("\n", i)
            i = n if end < 0 else end
            removed += 1
        else:
            out.append(ch)
            i += 1
    lines = [ln.rstrip() for ln in "".join(out).splitlines()]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n")
    return text + "\n", removed


def strip_makefile_comments(source: str) -> tuple[str, int]:
    """Remove comments from a Makefile, leaving ``#`` inside quotes alone."""
    out: list[str] = []
    removed = 0
    for line in source.splitlines():
        if line.lstrip().startswith("#"):
            removed += 1
            continue
        quote = ""
        cut = None
        for idx, ch in enumerate(line):
            if quote:
                quote = "" if ch == quote else quote
            elif ch in "\"'":
                quote = ch
            elif ch == "#" and idx > 0 and line[idx - 1] in " \t":
                cut = idx
                break
        if cut is not None:
            line = line[:cut].rstrip()
            removed += 1
        out.append(line.rstrip())
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip("\n")
    return text + "\n", removed


class _PythonMasker(ast.NodeTransformer):
    """Rename user-defined identifiers in Python code and drop docstrings."""

    def __init__(self, renamer: _Renamer, keep: set[str], local_args: set[str]) -> None:
        self.renamer = renamer
        self.keep = keep
        self.local_args = local_args
        self.removed = 0

    def _name(self, name: str) -> str:
        return name if name in self.keep else self.renamer.identifier(name)

    def _drop_docstring(self, node: ast.AST) -> None:
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]
            self.removed += 1
            if not body:
                body.append(ast.Pass())

    def visit_Module(self, node: ast.Module) -> ast.AST:  # noqa: N802
        self._drop_docstring(node)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
        self._drop_docstring(node)
        node.name = self._name(node.name)
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:  # noqa: N802
        self._drop_docstring(node)
        node.name = self._name(node.name)
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:  # noqa: N802
        self._drop_docstring(node)
        node.name = self._name(node.name)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        node.id = self._name(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:  # noqa: N802
        node.arg = self._name(node.arg)
        return self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> ast.AST:  # noqa: N802
        if node.arg and node.arg in self.local_args:
            node.arg = self._name(node.arg)
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        if isinstance(node.value, str):
            node.value = self.renamer.kernel_only(node.value)
        return node


def _mask_python(sources: list[str], renamer: _Renamer, report: MaskReport) -> list[str]:
    """Mask the code cells of a notebook, which share one namespace."""
    trees: list[ast.Module | None] = []
    keep = set(dir(builtins))
    local_args: set[str] = set()
    for src in sources:
        tree: ast.Module | None
        try:
            tree = ast.parse(src)
        except SyntaxError:
            trees.append(None)
            continue
        trees.append(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                for alias in node.names:
                    keep.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.arg):
                local_args.add(node.arg)
    out: list[str] = []
    for src, tree in zip(sources, trees, strict=True):
        if tree is None:
            report.notes.append(
                "a code cell could not be parsed; only its kernel name and "
                "whole-line comments were handled"
            )
            kept = [ln for ln in src.splitlines() if not ln.lstrip().startswith("#")]
            report.comments_removed += len(src.splitlines()) - len(kept)
            out.append(renamer.kernel_only("\n".join(kept)))
            continue
        masker = _PythonMasker(renamer, keep, local_args)
        new_tree = ast.fix_missing_locations(masker.visit(tree))
        report.comments_removed += masker.removed + _count_python_comments(src)
        out.append(ast.unparse(new_tree))
    return out


def _count_python_comments(source: str) -> int:
    import io
    import tokenize

    try:
        return sum(
            1
            for tok in tokenize.generate_tokens(io.StringIO(source).readline)
            if tok.type == tokenize.COMMENT
        )
    except (tokenize.TokenError, IndentationError):
        return 0


def _mask_notebook(path: Path, dest: Path, renamer: _Renamer, report: MaskReport) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    code = [c for c in cells if c["cell_type"] == "code"]
    sources = ["".join(c["source"]) for c in code]
    for cell, new_source in zip(code, _mask_python(sources, renamer, report), strict=True):
        cell["source"] = new_source.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
    new_cells: list[dict] = []
    markdown_seen = 0
    for cell in cells:
        if cell["cell_type"] == "markdown":
            markdown_seen += 1
            if markdown_seen > 1:
                continue
            cell = {**cell, "source": [NEUTRAL_SENTENCE]}
        new_cells.append(cell)
    if markdown_seen == 0:
        new_cells.insert(0, {"cell_type": "markdown", "metadata": {}, "source": [NEUTRAL_SENTENCE]})
    report.markdown_cells_replaced += markdown_seen
    notebook["cells"] = new_cells
    dest.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _choose_prefix(problem_dir: Path) -> str:
    corpus = ""
    for path in problem_dir.rglob("*"):
        if path.is_file() and path.suffix not in _SKIP_SUFFIXES:
            corpus += path.read_text(encoding="utf-8", errors="ignore")
    for prefix in ("mk", "nx", "qz", "zz"):
        if not re.search(rf"(?<![A-Za-z0-9_]){prefix}\d+(?![A-Za-z0-9_])", corpus):
            return prefix
    return "zz_"


def mask_problem(problem_dir: Path, dest: Path, alias: str) -> MaskReport:
    """Write a masked copy of ``problem_dir`` to ``dest`` and report every change.

    ``alias`` must be a plain identifier that is not itself in the word list. ``dest`` is
    emptied first if it exists. The kernel name is taken from the problem's notebook.
    """
    problem_dir, dest = Path(problem_dir), Path(dest)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", alias):
        raise ValueError(f"alias {alias!r} must be a plain identifier")
    if alias.lower() in WORD_LIST:
        raise ValueError(f"alias {alias!r} is itself in the word list; choose a neutral one")
    notebooks = sorted(problem_dir.glob("*.ipynb"))
    if len(notebooks) != 1:
        raise ValueError(f"{problem_dir} must contain exactly one notebook")
    kernel = notebooks[0].stem

    renamer = _Renamer(kernel, alias, _choose_prefix(problem_dir))
    report = MaskReport(kernel=kernel, alias=alias)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for source in sorted(problem_dir.rglob("*")):
        relative = source.relative_to(problem_dir)
        if source.is_dir() or _SKIP_DIRS & set(relative.parts) or source.suffix in _SKIP_SUFFIXES:
            continue
        renamer.current_file = str(relative)
        new_parts = [renamer.text(part) for part in relative.parts]
        target = dest.joinpath(*new_parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if str(relative) != str(Path(*new_parts)):
            report.renamed_files.append((str(relative), str(Path(*new_parts))))
        renamer.current_file = str(relative)

        if source.suffix == ".ipynb":
            _mask_notebook(source, target, renamer, report)
            continue
        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            shutil.copy2(source, target)
            continue
        if source.suffix in _C_SUFFIXES:
            text, removed = strip_c_comments(text)
            text = renamer.c_source(text)
        elif source.name == "Makefile":
            text, removed = strip_makefile_comments(text)
            text = renamer.text(text, words=False)
        else:
            removed = 0
            text = renamer.kernel_only(text)
        report.comments_removed += removed
        target.write_text(text, encoding="utf-8")
        if os_executable(source):
            target.chmod(source.stat().st_mode)

    for (file, original, replacement), count in sorted(renamer.counts.items()):
        report.substitutions.append(Substitution(file, original, replacement, count))
    log.info("masked %s as %s: %d substitutions", kernel, alias, len(report.substitutions))
    return report


def os_executable(path: Path) -> bool:
    """Return whether the owner may execute the file."""
    return bool(path.stat().st_mode & 0o100)
