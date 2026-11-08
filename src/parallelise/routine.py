"""The declaration of a routine, and what follows from it.

A notebook declares its routine in a dictionary named ROUTINE, beside an ordinary Python
function that computes the answer and a function that builds inputs of any size:

    def prefix_sum(x):
        return np.cumsum(x)

    def make_inputs(size, rng):
        return {"x": rng.uniform(-0.5, 0.5, size)}

    ROUTINE = {"function": prefix_sum, "make_inputs": make_inputs,
               "arguments": {"x": "float64[n]"}, "results": {"out": "float64[n]"},
               "test_sizes": [1, 1000, 200_000], "benchmark_size": 20_000_000,
               "tolerance": 1e-9}

Every argument and result has a type (float64, float32, int64 or int32) and, if it is an
array, a shape made of size names and whole numbers, such as float64[n, 3]. A size name is an
integer argument or is read from the shape of an array argument. A result written int64[<=n]
is an array of at most n elements whose length the routine decides. A result with the same
name as an argument is that argument, updated in place. The tolerance is relative to the
largest magnitude of the trusted result; a result may instead be given an absolute tolerance,
as ("absolute", 0.01). An optional "c_order" lists the parameter names in the order the C
routine takes them.

From the declaration follow the C prototype, the binary files in which the harness hands a
routine its inputs and gets back its results, and the comparison of two sets of results.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ParalleliseError

C_TYPES = {"float64": "double", "float32": "float", "int64": "long", "int32": "int"}

_VALUE = re.compile(r"^(float64|float32|int64|int32)(?:\[(.*)\])?$")
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
#: The C library's mathematical functions that a routine may call, each also in its float and
#: long double forms, and its other functions that a routine may call.
C_MATHS = (
    "sin cos tan asin acos atan atan2 sinh cosh tanh asinh acosh atanh exp exp2 expm1 log log2 "
    "log10 log1p sqrt cbrt pow hypot fabs floor ceil round lround llround trunc rint lrint "
    "nearbyint fmod remainder fmin fmax fma copysign erf erfc tgamma lgamma frexp ldexp modf "
    "sincos"
).split()
C_LIBRARY = (
    "malloc calloc realloc free aligned_alloc posix_memalign memcpy memmove memset memcmp qsort "
    "abs labs llabs rand srand rand_r printf fprintf puts fputs fputc putchar fwrite fflush perror "
    "stdout stderr abort exit __stack_chk_fail __assert_fail"
).split()
C_NAMES = set(C_LIBRARY) | {f"{m}{s}" for m in C_MATHS for s in ("", "f", "l")}

# Names that C reserves, that the C library uses, or that the generated driver uses for itself,
# cannot be declared.
_RESERVED = C_NAMES | set(
    "auto break case char const continue default do double else enum extern float for goto if "
    "inline int long register restrict return short signed sizeof static struct switch typedef "
    "union unsigned void volatile while bool true false main".split()
)


def _usable(name: str) -> bool:
    return bool(_NAME.match(name)) and name not in _RESERVED and not name.startswith("drv_")


def _literal(symbol: str) -> bool:
    """True for a dimension given as a whole number, such as the 3 of float64[n, 3]."""
    return symbol.isascii() and symbol.isdigit() and int(symbol) > 0


class DeclarationError(ParalleliseError):
    """A ROUTINE dictionary that cannot be used."""


@dataclass(frozen=True)
class Value:
    """One argument or result: its name, type and shape (empty for a number)."""

    name: str
    dtype: str
    shape: tuple[str, ...] = ()
    bounded: bool = False  # True for [<=n]: at most n elements, the routine says how many

    @property
    def is_array(self) -> bool:
        return bool(self.shape)


@dataclass(frozen=True)
class Declaration:
    """A checked ROUTINE dictionary, without its two functions."""

    name: str
    arguments: tuple[Value, ...]
    results: tuple[Value, ...]
    test_sizes: tuple[int, ...]
    benchmark_size: int
    tolerance: Mapping[str, tuple[str, float]]
    c_order: tuple[str, ...]

    @property
    def sizes(self) -> tuple[str, ...]:
        """Every size name, in the order of its first appearance."""
        seen: list[str] = []
        for value in (*self.arguments, *self.results):
            seen.extend(s for s in value.shape if s not in seen and not _literal(s))
        return tuple(seen)

    @property
    def updated(self) -> set[str]:
        """The arguments that the routine updates in place."""
        return {r.name for r in self.results} & {a.name for a in self.arguments}

    def returns_value(self) -> bool:
        """True when the C routine returns its one result, a number, instead of writing it."""
        return len(self.results) == 1 and not self.results[0].is_array and not self.updated

    def renamed(self, names: Mapping[str, str]) -> Declaration:
        """The same declaration under other names, such as those a masked run shows an agent.

        ``names`` must give a new name for the routine and for every argument, result and size.
        """

        def value(v: Value) -> Value:
            return replace(v, name=names[v.name], shape=tuple(names.get(s, s) for s in v.shape))

        return replace(self, name=names[self.name], arguments=tuple(map(value, self.arguments)),
                       results=tuple(map(value, self.results)),
                       tolerance={names[k]: t for k, t in self.tolerance.items()},
                       c_order=tuple(names[n] for n in self.c_order))  # fmt: skip


def parse_value(name: str, text: object) -> Value:
    """Read one entry such as "float64[n, n]" or "int64[<=n]"."""
    match = _VALUE.match(str(text).replace(" ", ""))
    if not _usable(name):
        raise DeclarationError(
            f"{name!r} cannot be the name of an argument or a result: C or its library uses "
            "it, or it begins with drv_, which the generated driver keeps for itself."
        )
    if match is None:
        raise DeclarationError(
            f"The entry {name!r}: {text!r} is not a type such as 'float64' or 'float64[n, m]'."
        )
    dtype, dims = match.group(1), match.group(2)
    if dims is None:
        return Value(name, dtype)
    bounded = dims.startswith("<=")
    shape = tuple(dims.removeprefix("<=").split(","))
    if not all(_usable(s) or _literal(s) for s in shape) or (bounded and len(shape) != 1):
        raise DeclarationError(
            f"The shape of {name!r} ({text!r}) is not a list of size names and whole numbers."
        )
    return Value(name, dtype, shape, bounded)


def parse_declaration(routine: Mapping[str, Any], name: str) -> Declaration:
    """Check a ROUTINE dictionary, read without its functions, and return it."""
    if not _usable(name):
        raise DeclarationError(
            f"The function {name!r} cannot become a C routine of that name: C or its library "
            "uses it, it is not a plain name, or it begins with drv_. Rename the function."
        )
    try:
        arguments = tuple(parse_value(k, v) for k, v in dict(routine["arguments"]).items())
        results = tuple(parse_value(k, v) for k, v in dict(routine["results"]).items())
        test_sizes = tuple(int(s) for s in routine["test_sizes"])
        benchmark_size = int(routine["benchmark_size"])
        raw_tol = routine.get("tolerance")
    except (KeyError, TypeError, ValueError) as exc:
        raise DeclarationError(
            f"ROUTINE needs arguments, results, test_sizes and benchmark_size, and a tolerance "
            f"for any result that is not an integer ({exc})."
        ) from exc
    if not results or not test_sizes or min(test_sizes) < 1 or benchmark_size < 1:
        raise DeclarationError("ROUTINE needs at least one result and sizes of 1 or more.")
    tolerance = {r.name: _tolerance(raw_tol, r) for r in results}
    integer_arguments = {a.name for a in arguments if a.dtype.startswith("int") and not a.is_array}
    shaped = {s for a in arguments for s in a.shape}
    for value in results:
        loose = [s for s in value.shape if not _literal(s) and s not in shaped | integer_arguments]
        if loose:
            raise DeclarationError(
                f"The size {loose[0]!r} of the result {value.name!r} is neither an integer "
                "argument nor the size of an array argument."
            )
    for value in arguments:
        if value.name in {r.name for r in results}:
            updated = next(r for r in results if r.name == value.name)
            if (updated.dtype, updated.shape) != (value.dtype, value.shape):
                raise DeclarationError(
                    f"{value.name!r} is updated in place, so it must have the same type and "
                    "shape as an argument and as a result."
                )
    declaration = Declaration(name, arguments, results, test_sizes, benchmark_size, tolerance, ())
    default = default_order(declaration)
    order = tuple(routine.get("c_order", default))
    if sorted(order) != sorted(default):
        raise DeclarationError(
            f"c_order {list(order)} must list exactly these names: {sorted(default)}."
        )
    return Declaration(name, arguments, results, test_sizes, benchmark_size, tolerance, order)


def _tolerance(raw: object, result: Value) -> tuple[str, float]:
    """A result's tolerance; integers are always compared exactly, so they need none."""
    value = raw.get(result.name) if isinstance(raw, Mapping) else raw
    if isinstance(value, (tuple, list)) and len(value) == 2 and value[0] == "absolute":
        return ("absolute", float(value[1]))
    if isinstance(value, (int, float)) and value >= 0:
        return ("relative", float(value))
    if value is None and result.dtype.startswith("int"):
        return ("relative", 0.0)
    raise DeclarationError(
        f"The tolerance for {result.name!r} must be a number, a fraction of the trusted result's "
        f"largest magnitude, or ('absolute', number), not {value!r}. Integer results are "
        "compared exactly and need none; one number may serve every result."
    )


def default_order(declaration: Declaration) -> tuple[str, ...]:
    """Arguments, then results not updated in place, then the sizes that are not arguments."""
    names = [a.name for a in declaration.arguments]
    if not declaration.returns_value():
        names += [r.name for r in declaration.results if r.name not in declaration.updated]
    names += [s for s in declaration.sizes if s not in names]
    return tuple(names)


# --- the C prototype -------------------------------------------------------------------------


def c_prototype(declaration: Declaration, name: str | None = None) -> str:
    """Return the C prototype, for example 'void scan(const double *x, double *out, long n)'."""
    arguments = {a.name: a for a in declaration.arguments}
    results = {r.name: r for r in declaration.results}
    parameters = []
    for item in declaration.c_order:
        if item in arguments:
            value = arguments[item]
            const = "" if item in declaration.updated else "const "
            ctype = C_TYPES[value.dtype]
            parameters.append(f"{const}{ctype} *{item}" if value.is_array else f"{ctype} {item}")
        elif item in results:
            value = results[item]
            parameters.append(f"{C_TYPES[value.dtype]} *{item}")
            if value.bounded:
                parameters.append(f"long *{item}_count")
        else:
            parameters.append(f"long {item}")
    returned = C_TYPES[declaration.results[0].dtype] if declaration.returns_value() else "void"
    return f"{returned} {name or declaration.name}({', '.join(parameters)})"


# --- the files that carry inputs and results -------------------------------------------------
#
# An input file holds every size, as a 64-bit integer, in the order of Declaration.sizes, and
# then every argument in declared order. A result file holds every result in declared order;
# a bounded result is preceded by its length. Numbers are in the machine's own byte order.


def extents(value: Value, sizes: Mapping[str, int]) -> tuple[int, ...]:
    """The length of each of a value's dimensions, given the sizes."""
    return tuple(int(s) if _literal(s) else sizes[s] for s in value.shape)


def sizes_of(declaration: Declaration, inputs: Mapping[str, Any]) -> dict[str, int]:
    """Work out every size from the inputs, and check that the arrays agree about them."""
    found: dict[str, int] = {}
    for value in declaration.arguments:
        if value.dtype.startswith("int") and not value.is_array and value.name in declaration.sizes:
            found[value.name] = int(inputs[value.name])
    for value in declaration.arguments:
        if not value.is_array:
            continue
        shape = np.shape(inputs[value.name])
        if len(shape) != len(value.shape):
            raise DeclarationError(
                f"The input {value.name!r} has {len(shape)} dimensions, but its declared shape "
                f"{list(value.shape)} has {len(value.shape)}."
            )
        for symbol, extent in zip(value.shape, shape, strict=True):
            if _literal(symbol):
                if extent != int(symbol):
                    raise DeclarationError(
                        f"The input {value.name!r} has shape {shape}, but its declared shape is "
                        f"[{', '.join(value.shape)}]."
                    )
            elif found.setdefault(symbol, extent) != extent:
                raise DeclarationError(
                    f"The input {value.name!r} makes the size {symbol} {extent}, but another "
                    f"input makes it {found[symbol]}."
                )
    return found


def write_inputs(path: Path, declaration: Declaration, inputs: Mapping[str, Any]) -> dict[str, int]:
    """Write one call's inputs to a file for the C driver; return the sizes."""
    sizes = sizes_of(declaration, inputs)
    with open(path, "wb") as handle:
        np.array([sizes[s] for s in declaration.sizes], dtype=np.int64).tofile(handle)
        for value in declaration.arguments:
            np.ascontiguousarray(inputs[value.name], dtype=value.dtype).tofile(handle)
    return sizes


def read_sizes(path: Path, declaration: Declaration) -> dict[str, int]:
    """Read the sizes at the start of an input file."""
    count = len(declaration.sizes)
    with open(path, "rb") as handle:
        values = np.fromfile(handle, np.int64, count).tolist()
    return dict(zip(declaration.sizes, values, strict=True))


def read_inputs(path: Path, declaration: Declaration) -> dict[str, Any]:
    """Read an input file back into the values the Python function is called with."""
    sizes = read_sizes(path, declaration)
    inputs: dict[str, Any] = {}
    with open(path, "rb") as handle:
        handle.seek(8 * len(declaration.sizes))
        for value in declaration.arguments:
            shape = extents(value, sizes)
            data = np.fromfile(handle, value.dtype, math.prod(shape))
            inputs[value.name] = data.reshape(shape) if value.is_array else data[0].item()
    return inputs


def write_results(path: Path, declaration: Declaration, results: Mapping[str, Any]) -> None:
    """Write results in the form the C driver writes them."""
    with open(path, "wb") as handle:
        for value in declaration.results:
            data = np.ascontiguousarray(results[value.name], dtype=value.dtype)
            if value.bounded:
                np.array([data.size], dtype=np.int64).tofile(handle)
            data.tofile(handle)


def read_results(path: Path, declaration: Declaration, sizes: Mapping[str, int]) -> dict[str, Any]:
    """Read a result file; raise DeclarationError if it is short or too long."""
    results: dict[str, Any] = {}
    with open(path, "rb") as handle:
        for value in declaration.results:
            shape = extents(value, sizes)
            if value.bounded:
                length = np.fromfile(handle, np.int64, 1)
                if length.size != 1 or not 0 <= length[0] <= shape[0]:
                    raise DeclarationError(
                        f"The result file {path} gives {value.name} a length outside 0 to "
                        f"{shape[0]}."
                    )
                shape = (int(length[0]),)
            data = np.fromfile(handle, value.dtype, math.prod(shape))
            if data.size != math.prod(shape):
                raise DeclarationError(f"The result file {path} is shorter than the declaration.")
            results[value.name] = data.reshape(shape) if value.is_array else data[0]
        if handle.read(1):
            raise DeclarationError(f"The result file {path} is longer than the declaration.")
    return results


def as_results(declaration: Declaration, returned: Any) -> dict[str, Any]:
    """Match what the Python function returned with the declared results.

    A function with one result returns it; one with several returns them in order, as a tuple
    or a list, or as a dictionary keyed by name.
    """
    names = [r.name for r in declaration.results]
    if isinstance(returned, Mapping):
        values = [returned.get(n) for n in names]
    elif len(names) == 1:
        values = [returned]
    elif isinstance(returned, (tuple, list)) and len(returned) == len(names):
        values = list(returned)
    else:
        raise DeclarationError(
            f"The function returned {type(returned).__name__}, but the declaration lists "
            f"{len(names)} results: {names}."
        )
    return dict(zip(names, values, strict=True))


# --- comparing results -------------------------------------------------------------------------


def compare(declaration: Declaration, got: Mapping[str, Any], want: Mapping[str, Any]) -> list[str]:
    """Return what is wrong with ``got`` against the trusted ``want``; empty when they agree.

    Each result must have the declared kind of number, the trusted result's shape and finite
    values. Integers must be equal. Real numbers must agree within the result's tolerance:
    a relative tolerance is a fraction of the largest magnitude in the trusted result.
    """
    problems = []
    for value in declaration.results:
        a, b = np.asarray(got[value.name]), np.asarray(want[value.name])
        integer = value.dtype.startswith("int")
        if (a.dtype.kind in "iu") != integer:
            problems.append(f"{value.name} holds {a.dtype} values, not {value.dtype}")
        elif a.shape != b.shape:
            problems.append(f"{value.name} has shape {a.shape}, not {b.shape}")
        elif not integer and not np.all(np.isfinite(a)):
            problems.append(f"{value.name} holds a value that is not a finite number")
        elif integer and not np.array_equal(a, b):
            problems.append(f"{value.name} differs in {int(np.sum(a != b))} places")
        elif not integer and a.size:
            kind, tol = declaration.tolerance[value.name]
            scale = float(np.max(np.abs(b))) if kind == "relative" else 1.0
            error = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            if error > tol * (scale or 1.0):
                limit = "of its largest magnitude" if kind == "relative" else "absolute"
                problems.append(
                    f"{value.name} differs by up to {error:.3g}; the tolerance is {tol:g} {limit}"
                )
    return problems
