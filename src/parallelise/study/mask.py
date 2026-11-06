"""Masking what an agent sees of a routine, so that a model cannot recognise the algorithm by name.

A language model may have seen well-known algorithms many times, so a result on a problem called
"jacobi" may reflect memory rather than ability. ``mask_sources`` masks the three things an agent
reads: the notebook's Python, the serial C and the generated header. Every number and every line
of logic stay as they were; what changes is this:

- the routine's name becomes the alias;
- every other name that the sources define, from the arguments and results to the local
  variables, helper functions and constants, becomes a neutral name such as ``v7``, the same
  for the same original name in all three sources;
- comments and Python docstrings are removed, except the header's, which states the rules.

Names that the sources do not define are left alone, because they belong to Python, NumPy, the C
library or the backend. In C, what the system headers declare is found by asking the compiler;
on OpenMP pragma lines only the text inside parentheses is renamed.
"""

from __future__ import annotations

import ast
import re
import subprocess

from ..errors import ParalleliseError

_IDENT = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*")
# What C itself and its preprocessor reserve; names beginning with two underscores belong to the
# compiler and the library.
_C_WORDS = set(
    "auto break case char const continue default do double else enum extern float for goto if "
    "inline int long register restrict return short signed sizeof static struct switch typedef "
    "union unsigned void volatile while _Bool _Complex include define undef ifdef ifndef elif "
    "endif defined pragma error line".split()
)
_C_LITERAL_OR_COMMENT = re.compile(
    r'("(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|/\*.*?\*/|//[^\n]*)', re.DOTALL)


class _Renamer:
    """Gives each name that the sources define a neutral name, the same in every source."""

    def __init__(self, routine: str, alias: str, taken: set[str]) -> None:
        self.names = {routine: alias}
        self.alias, self.taken, self.count = alias, taken, 0

    def __call__(self, name: str) -> str:
        if name in (self.alias, f"{self.alias.upper()}_H"):
            return name  # the alias, and its header's guard
        if name not in self.names:
            self.count += 1
            while f"v{self.count}" in self.taken:
                self.count += 1
            self.names[name] = f"v{self.count}"
        return self.names[name]


def _system_names(c_source: str) -> set[str]:
    """The names that the system headers which ``c_source`` includes declare or define.

    These are their macros, functions, types, tags and variables, read from the compiler's
    expansion of the headers; the parameters of their prototypes and the bodies of their
    inline functions are not counted, because those names belong to nobody.
    """
    includes = "\n".join(line for line in c_source.splitlines()
                         if re.match(r"\s*#\s*include\s*<", line))  # fmt: skip
    try:
        listed = subprocess.run(["gcc", "-E", "-dD", "-x", "c", "-"], input=includes,
                                capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ParalleliseError(f"Masking needs gcc to list the system's names ({exc}).") from exc
    macros = re.findall(r"^#define\s+(\w+)", listed, re.MULTILINE)
    flat = re.sub(r"^#.*$", "", listed, flags=re.MULTILINE)
    while "{" in flat:  # leave out every body, innermost first
        shorter = re.sub(r"\{[^{}]*\}", " ", flat)
        if shorter == flat:
            break
        flat = shorter
    functions = re.findall(r"\b(\w+)\s*\(", flat)
    tags = re.findall(r"\b(?:struct|union|enum)\s+(\w+)", flat)
    last_of_each = re.findall(r"\b(\w+)\s*(?:\[[^\]]*\]\s*)*;", flat)
    return set(macros + functions + tags + last_of_each)


def _mask_c(source: str, rename: _Renamer, kept: set[str]) -> str:
    """Rename the names a C source defines, leaving its literals, comments and directives' words."""
    source = re.sub(r'(#\s*include\s*)"(\w+)\.h"',
                    lambda m: f'{m.group(1)}"{rename(m.group(2))}.h"', source)

    def names(text: str) -> str:
        return _IDENT.sub(lambda m: m.group(0) if m.group(0) in kept
                          or m.group(0).startswith("__") else rename(m.group(0)), text)

    parts = _C_LITERAL_OR_COMMENT.split(source)
    for i in range(0, len(parts), 2):  # the code between the literals and comments
        lines = []
        for line in parts[i].splitlines(keepends=True):
            if re.match(r"\s*#\s*pragma", line):
                line = re.sub(r"\([^()]*\)", lambda m: names(m.group(0)), line)
            elif not re.match(r"\s*#\s*include", line):
                line = names(line)
            lines.append(line)
        parts[i] = "".join(lines)
    return "".join(parts)


def strip_c_comments(source: str) -> str:
    """Remove the comments from C source, keeping string and character literals."""
    text = _C_LITERAL_OR_COMMENT.sub(lambda m: " " if m.group(0)[:2] in ("/*", "//")
                                     else m.group(0), source)  # fmt: skip
    lines = [line.rstrip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n") + "\n"


class _PythonMasker(ast.NodeTransformer):
    """Renames the names that Python code defines, and drops its docstrings."""

    def __init__(self, rename: _Renamer, routine: str, defined: set[str],
                 functions: set[str]) -> None:
        self.rename, self.routine = rename, routine
        self.defined, self.functions = defined, functions

    def _name(self, name: str) -> str:
        return self.rename(name) if name in self.defined else name

    def _drop_docstring(self, node: ast.AST) -> None:
        body = getattr(node, "body", [])
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            del body[0]
            body[:] = body or [ast.Pass()]

    def visit_Module(self, node: ast.Module) -> ast.AST:  # noqa: N802
        self._drop_docstring(node)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
        self._drop_docstring(node)
        node.name = self._name(node.name)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:  # noqa: N802
        node.id = self._name(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:  # noqa: N802
        node.arg = self._name(node.arg)
        return self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> ast.AST:  # noqa: N802
        node.names = [self._name(n) for n in node.names]
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        # The keywords of a call to the notebook's own function name its parameters.
        if isinstance(node.func, ast.Name) and node.func.id in self.functions:
            for keyword in node.keywords:
                keyword.arg = self._name(keyword.arg) if keyword.arg else None
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        if isinstance(node.value, str):
            node.value = re.sub(rf"(?<![A-Za-z0-9]){re.escape(self.routine)}(?![A-Za-z0-9])",
                                self.rename.alias, node.value, flags=re.IGNORECASE)  # fmt: skip
        return node


def _mask_python(source: str, rename: _Renamer, routine: str) -> str:
    tree = ast.parse(source)
    imported = {(a.asname or a.name).split(".")[0] for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom)) for a in node.names}
    functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    defined = functions | {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)} | {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    masker = _PythonMasker(rename, routine, defined - imported, functions)
    return ast.unparse(ast.fix_missing_locations(masker.visit(tree))) + "\n"


def mask_sources(python: str, serial_c: str, header: str, routine: str,
                 alias: str) -> tuple[str, str, str]:
    """Mask a routine's Python, its serial C and its header together, and return all three.

    The header must already be made under the alias. ``alias`` must be a plain name that the
    sources do not use, or ParalleliseError is raised.
    """
    used = set(_IDENT.findall(python + serial_c))
    kept = _system_names(serial_c) | _C_WORDS
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", alias) or alias in used | kept:
        raise ParalleliseError(f"The alias {alias!r} must be a plain name, of letters, digits and "
                               "underscores, that the sources do not use already, such as "
                               "problem_a.")  # fmt: skip
    rename = _Renamer(routine, alias, used | kept | set(_IDENT.findall(header)))
    masked_header = _mask_c(header, rename, kept)
    masked_python = _mask_python(python, rename, routine)
    return masked_python, _mask_c(strip_c_comments(serial_c), rename, kept), masked_header
