"""The call recorder: a small trusted C file that watches the benchmark's calls to the routine.

The trusted benchmark times the agent's routine and prints a table, but on its own it never
looks at the routine's answers, and it takes its one-worker baseline from the routine itself.
A routine can exploit both. It can be right at the sizes the test uses and wrong at the
benchmark's larger size; it can be serial and make its own one-worker run slow on purpose;
or it can keep its answer from one call to the next and simply wait for a believable time.

The check closes these routes without changing the benchmark. It writes a C file from the
routine's signature in the problem's header, compiles it, and links it into the benchmark
with the linker's ``--wrap`` option, so that every call the benchmark makes to the routine
goes through the recorder first. The recorder works in one of two ways, chosen by the
environment of the benchmark run:

- When ``PARALLELISE_CALL_TIMES`` names a file, it records, for each call, the number of
  threads, the wall time and the processor time, which is the time summed over every thread
  of the process. Processor time divided by wall time is the number of processors the call
  kept busy on average. A serial routine keeps about one busy, however its table looks, and
  a routine that waits keeps almost none.
- When ``PARALLELISE_CALL_VERIFY`` names a file, it runs the problem's known-good routine
  beside the agent's on each call, on an identical copy of the arguments, and records how
  far apart the two answers are.

Arrays are copied by their size as the memory allocator reports it, so the recorder assumes
that the benchmark passes arrays it obtained from ``malloc``, as every shipped benchmark does.
The recorder is for OpenMP routines; MPI and CUDA benchmarks are not watched.

This module writes the C file and reads what the recorder wrote. The check builds and runs
it (see ``check.py``).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: The environment variable that asks the recorder to time each call, naming its output file.
TIMES_VARIABLE = "PARALLELISE_CALL_TIMES"
#: The environment variable that asks the recorder to compare each call's answer with the
#: known-good routine's, naming its output file.
VERIFY_VARIABLE = "PARALLELISE_CALL_VERIFY"

#: The file names the check uses in the judging folder.
SOURCE_NAME = "parallelise_recorder.c"
OBJECT_NAME = "parallelise_recorder.o"
KNOWN_GOOD_SOURCE = "parallelise_known_good.c"
KNOWN_GOOD_OBJECT = "parallelise_known_good.o"
TIMES_FILE = "calls-timed.csv"
VERIFY_FILE = "calls-verified.csv"
VERIFY_TABLE = "bench-verify.csv"

#: Two floating-point answers agree when the largest difference between them is at most this
#: fraction of the largest value in the known-good answer. Parallel code adds numbers in a
#: different order from serial code, so the last few digits may differ, but no more.
RELATIVE_TOLERANCE = 1.0e-6

#: A row of the table whose speed-up is more than this many times the processors its call kept
#: busy is credited with only the busy processors. The allowance covers noise in the one-worker
#: time, which can make an honest row look up to about a tenth faster than its busy processors
#: allow. It must stay below the lowest floor in use, 1.25, so that a serial routine, which
#: keeps one processor busy, can never be credited with a useful speed-up.
BUSY_ALLOWANCE = 1.2

#: How closely the recorder's wall time for a call must match the time in the table: within
#: this fraction of the table's time, plus ``WALL_SLACK_S`` seconds.
WALL_TOLERANCE = 0.1
WALL_SLACK_S = 0.002


def known_good_symbol(name: str) -> str:
    """Return the name under which the known-good routine is linked beside the agent's."""
    return f"parallelise_known_good_{name}"


@dataclass(frozen=True)
class Parameter:
    """One parameter of the routine: its C type, its name, and what the recorder does with it."""

    ctype: str
    name: str

    @property
    def pointer(self) -> bool:
        return "*" in self.ctype

    @property
    def writable(self) -> bool:
        """True for a pointer through which the routine may write, so its target is an answer."""
        return self.pointer and not self.ctype.split("*", 1)[0].split().count("const")

    @property
    def element(self) -> str:
        """The type that a pointer points to, without ``const``: ``double`` for ``double *x``."""
        words = self.ctype.split("*", 1)[0].split()
        return " ".join(w for w in words if w != "const")


@dataclass(frozen=True)
class Prototype:
    """The signature of the routine, read from the problem's header."""

    name: str
    returns: str
    parameters: tuple[Parameter, ...]

    def declaration(self, name: str) -> str:
        params = ", ".join(f"{p.ctype} {p.name}" for p in self.parameters) or "void"
        return f"{self.returns} {name}({params})"


class PrototypeError(ValueError):
    """The routine's signature could not be read from the header in a form the recorder handles."""


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", " ", text)


def parse_prototype(header_text: str) -> Prototype:
    """Read the declaration of the routine from a C header and return it.

    The header must declare exactly one function, as every problem's header does. Its name
    need not be the routine's file name: ``cg.h``, for example, declares ``cgsolve``. Raises
    ``PrototypeError`` if there is not exactly one declaration, or if it uses a form the
    recorder does not handle: an array parameter, a function pointer or a variable number of
    arguments.
    """
    text = _strip_comments(header_text)
    pattern = (
        r"^[ \t]*((?:[A-Za-z_]\w*[ \t\*]+)+?)([A-Za-z_]\w*)[ \t]*\(([^()]*)\)[ \t]*;"
    )
    matches = [m for m in re.finditer(pattern, text, flags=re.MULTILINE)
               if not m.group(1).lstrip().startswith("#")]  # fmt: skip
    if len(matches) != 1:
        raise PrototypeError(
            f"the header declares {len(matches)} functions in a form that can be read, not one"
        )
    match = matches[0]
    name = match.group(2)
    returns = " ".join(match.group(1).replace("*", " * ").split()).replace(" *", "*")
    returns = returns.replace("extern ", "").strip()
    raw = match.group(3).strip()
    parameters: list[Parameter] = []
    if raw and raw != "void":
        for index, part in enumerate(raw.split(",")):
            part = " ".join(part.split())
            if not part or "[" in part or "(" in part or part == "...":
                raise PrototypeError(f"the parameter {part!r} of {name} cannot be handled")
            found = re.match(r"^(.*?[\s\*])([A-Za-z_]\w*)$", part)
            if found and found.group(1).strip() and found.group(2) not in _TYPE_WORDS:
                ctype, pname = found.group(1).strip(), found.group(2)
            else:
                ctype, pname = part, f"argument_{index}"
            parameters.append(Parameter(" ".join(ctype.replace("*", " * ").split()), pname))
    return Prototype(name, returns, tuple(parameters))


_TYPE_WORDS = {"int", "long", "short", "char", "double", "float", "unsigned", "signed", "size_t"}
_FLOATING = {"double", "float"}


def _compare_answer(p: Parameter, index: int) -> str:
    """Return the C statement that compares one array answer of the agent's with the known one."""
    if p.element in _FLOATING:
        return (
            f"pr_floating_{p.element.replace(' ', '_')}(after_{index}, {p.name}, "
            f"size_{index} / sizeof({p.element}), c);"
        )
    return f"if (memcmp(after_{index}, {p.name}, size_{index}) != 0) c->agrees = 0;"


def _block(indent: int, lines: Sequence[str]) -> str:
    pad = " " * indent
    return "".join(f"{pad}{line}\n" if line else "\n" for line in lines)


def recorder_source(prototype: Prototype, header: str) -> str:
    """Return the C source of the call recorder for ``prototype``, declared in ``header``."""
    name = prototype.name
    args = ", ".join(p.name for p in prototype.parameters)
    void = prototype.returns == "void"
    floating_return = prototype.returns in _FLOATING
    writable = [(i, p) for i, p in enumerate(prototype.parameters) if p.writable]
    got = "" if void else f"{prototype.returns} got = "
    want = "" if void else f"{prototype.returns} want = "
    give = "return;" if void else "return got;"

    verify: list[str] = ["int copied = 1;"]
    for i, p in writable:
        verify += [
            f"size_t size_{i} = {p.name} ? malloc_usable_size((void *){p.name}) : 0;",
            f"void *before_{i} = pr_copy({p.name}, size_{i});",
            f"if (size_{i} && !before_{i}) copied = 0;",
        ]
    verify.append(f"{got}__real_{name}({args});")
    for i, p in writable:
        verify += [
            f"void *after_{i} = pr_copy({p.name}, size_{i});",
            f"if (size_{i} && !after_{i}) copied = 0;",
            f"if (copied && size_{i}) memcpy({p.name}, before_{i}, size_{i});",
        ]
    verify += [
        "double k0 = pr_now(CLOCK_MONOTONIC);",
        f"{want}{known_good_symbol(name)}({args});",
        "double k1 = pr_now(CLOCK_MONOTONIC);",
        "if (c) {",
        "    c->width = width;",
        "    c->known_wall = k1 - k0;",
        "    c->checked = copied;",
        "    c->agrees = 1;",
        "    if (copied) {",
    ]
    verify += [f"        {_compare_answer(p, i)}" for i, p in writable]
    if floating_return:
        verify += [
            "        double got_value = (double)got, want_value = (double)want;",
            "        pr_floating_double(&got_value, &want_value, 1, c);",
        ]
    elif not void:
        verify.append("        if (memcmp(&got, &want, sizeof got) != 0) c->agrees = 0;")
    verify += [
        "        if (!(c->difference <= PR_RTOL * c->scale)) c->agrees = 0;",
        "    }",
        "}",
    ]
    for i, p in writable:
        verify += [
            f"if (copied && size_{i}) memcpy({p.name}, after_{i}, size_{i});",
            f"free(before_{i});",
            f"free(after_{i});",
        ]
    verify.append(give)

    timed = [
        "double w0 = pr_now(CLOCK_MONOTONIC), c0 = pr_now(CLOCK_PROCESS_CPUTIME_ID);",
        f"{got}__real_{name}({args});",
        "double c1 = pr_now(CLOCK_PROCESS_CPUTIME_ID), w1 = pr_now(CLOCK_MONOTONIC);",
        "if (c) {",
        "    c->width = width;",
        "    c->wall = w1 - w0;",
        "    c->cpu = c1 - c0;",
        "}",
        give,
    ]
    floating_types = {p.element for _, p in writable if p.element in _FLOATING}
    if floating_return:
        floating_types.add("double")
    head = (
        _HEAD.replace("@HEADER@", header)
        .replace("@RTOL@", repr(RELATIVE_TOLERANCE))
        .replace("@TIMES@", TIMES_VARIABLE)
        .replace("@VERIFY@", VERIFY_VARIABLE)
    )
    return (
        head
        + f"{prototype.declaration('__real_' + name)};\n"
        + f"{prototype.declaration('__wrap_' + name)};\n"
        + f"{prototype.declaration(known_good_symbol(name))};\n\n"
        + "".join(_FLOATING_COMPARE.replace("TYPE", t) for t in sorted(floating_types))
        + f"{prototype.declaration('__wrap_' + name)}\n{{\n"
        + _block(4, [
            "if (pr_mode < 0) pr_start();",
            "struct pr_call *c = NULL;",
            "if (pr_mode > 0) {",
            "    if (pr_count < PR_LIMIT) c = &pr_calls[pr_count];",
            "    pr_count++;",
            "}",
            "int width = omp_get_max_threads();",
            "if (pr_mode == 2) {",
        ])
        + _block(8, verify)
        + _block(4, ["}"])
        + _block(4, timed)
        + "}\n"
    )  # fmt: skip


_HEAD = r"""/* The call recorder, written by the check into the judging folder. It is linked into
 * the benchmark with the linker's --wrap option, so that each of the benchmark's calls to the
 * routine passes through it. The agent never sees this file. */
#define _GNU_SOURCE
#include <malloc.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "@HEADER@"

#define PR_LIMIT 4096
#define PR_RTOL @RTOL@

struct pr_call {
    int width, checked, agrees;
    double wall, cpu, known_wall, difference, scale;
};
static struct pr_call pr_calls[PR_LIMIT];
static int pr_count, pr_mode = -1;
static const char *pr_path;

static double pr_now(clockid_t clock)
{
    struct timespec ts;
    clock_gettime(clock, &ts);
    return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

static void pr_write(void)
{
    FILE *f = fopen(pr_path, "w");
    if (!f) return;
    fprintf(f, "call,width,wall_s,cpu_s,checked,agrees,difference,scale,known_wall_s\n");
    for (int i = 0; i < pr_count; i++) {
        if (i >= PR_LIMIT) {
            fprintf(f, "%d,overflow\n", i);
            break;
        }
        const struct pr_call *c = &pr_calls[i];
        fprintf(f, "%d,%d,%.9f,%.9f,%d,%d,%.17g,%.17g,%.9f\n", i, c->width, c->wall, c->cpu,
                c->checked, c->agrees, c->difference, c->scale, c->known_wall);
    }
    fclose(f);
}

static void pr_start(void)
{
    const char *verify = getenv("@VERIFY@");
    const char *times = getenv("@TIMES@");
    pr_mode = 0;
    if (verify && *verify) {
        pr_mode = 2;
        pr_path = verify;
    } else if (times && *times) {
        pr_mode = 1;
        pr_path = times;
    }
    if (pr_mode > 0) atexit(pr_write);
}

__attribute__((unused)) static void *pr_copy(const void *p, size_t n)
{
    if (!n) return NULL;
    void *copy = malloc(n);
    if (copy) memcpy(copy, p, n);
    return copy;
}

"""

_FLOATING_COMPARE = r"""static void pr_floating_TYPE(const void *got_p, const void *want_p,
                             size_t n, struct pr_call *c)
{
    const TYPE *got = got_p, *want = want_p;
    for (size_t i = 0; i < n; i++) {
        double w = (double)want[i], g = (double)got[i];
        if (isfinite(w) && fabs(w) > c->scale) c->scale = fabs(w);
        if (memcmp(&got[i], &want[i], sizeof(TYPE)) == 0) continue;
        if (!isfinite(g) || !isfinite(w)) {
            c->agrees = 0;
            continue;
        }
        if (fabs(g - w) > c->difference) c->difference = fabs(g - w);
    }
}

"""


@dataclass(frozen=True)
class Call:
    """One call that the recorder saw, in the order the benchmark made them."""

    call: int
    width: int
    wall_s: float
    cpu_s: float
    checked: bool
    agrees: bool
    difference: float
    scale: float
    known_wall_s: float

    @property
    def busy(self) -> float:
        """The number of processors the call kept busy on average: processor time over wall time."""
        return self.cpu_s / self.wall_s if self.wall_s > 0 else 0.0


def read_calls(path: Path) -> list[Call]:
    """Read the recorder's output file. Raises ``ValueError`` if it is malformed or overflowed."""
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("call,"):
        raise ValueError(f"{path.name} has no header line")
    calls: list[Call] = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) == 2 and parts[1] == "overflow":
            raise ValueError(f"the benchmark made more calls than the recorder can hold ({line})")
        if len(parts) != 9:
            raise ValueError(f"{path.name} has a malformed line: {line!r}")
        calls.append(
            Call(
                call=int(parts[0]), width=int(parts[1]), wall_s=float(parts[2]),
                cpu_s=float(parts[3]), checked=parts[4] == "1", agrees=parts[5] == "1",
                difference=float(parts[6]), scale=float(parts[7]),
                known_wall_s=float(parts[8]),
            )
        )  # fmt: skip
    return calls


def _threads(width: int) -> str:
    return "1 thread" if width == 1 else f"{width} threads"


def verification_problems(calls: Sequence[Call]) -> list[str]:
    """Return a description of each call whose answer differs from the known-good routine's."""
    if not calls:
        return ["the benchmark made no call to the routine that could be compared"]
    problems: list[str] = []
    for c in calls:
        if not c.checked:
            problems.append(
                f"call {c.call + 1} ({_threads(c.width)}) could not be compared, because there "
                "was not enough memory to copy its arrays"
            )
        elif not c.agrees:
            if math.isfinite(c.difference) and c.scale > 0:
                size = f"{c.difference:.3g} where the answer is of size {c.scale:.3g}"
            else:
                size = "a value that is not a finite number, or that differs exactly"
            problems.append(
                f"call {c.call + 1} ({_threads(c.width)}) differs from the known-good routine by "
                f"{size}"
            )
    return problems


@dataclass(frozen=True)
class Credit:
    """What the recorded calls allow the table to claim.

    ``best`` is the best speed-up the table is credited with. ``capped`` describes each row
    whose speed-up was more than its busy processors allow, and ``problems`` each row that
    the recorded calls do not support at all.
    """

    best: float | None
    capped: list[str]
    problems: list[str]


def credit_speedups(rows: Sequence[tuple[int, float, float]], calls: Sequence[Call]) -> Credit:
    """Compare the table's rows with the recorded calls, and return the speed-up they support.

    ``rows`` holds the width, the time and the speed-up of each row of the table. Each row is
    matched with the last call the recorder saw at that width, which is the timed call (an
    earlier call at the same width is the untimed warm-up). A row with no such call, or whose
    time is far from the call's wall time, did not come from these calls, and is a problem. A
    row whose speed-up is more than ``BUSY_ALLOWANCE`` times the processors its call kept busy
    is credited with only the number of busy processors, since work shared among that many
    processors cannot finish faster than that.
    """
    last: dict[int, Call] = {}
    for c in calls:
        last[c.width] = c
    capped: list[str] = []
    problems: list[str] = []
    best: float | None = None
    for width, time_s, speedup in rows:
        found = last.get(width)
        if found is None:
            problems.append(f"the table has a row for {width} threads, but the benchmark made "
                            f"no call to the routine with {width} threads")  # fmt: skip
            continue
        c = found
        if abs(c.wall_s - time_s) > WALL_TOLERANCE * time_s + WALL_SLACK_S:
            problems.append(
                f"the table gives {time_s:.6f} s for {width} threads, but the call took "
                f"{c.wall_s:.6f} s"
            )
            continue
        credited = speedup
        if width > 1 and speedup > BUSY_ALLOWANCE * c.busy:
            credited = c.busy
            capped.append(
                f"at {width} threads the table shows a speed-up of {speedup:.2f}, but the call "
                f"kept only {c.busy:.2f} processors busy on average, so it is credited with "
                f"{credited:.2f}"
            )
        best = credited if best is None else max(best, credited)
    return Credit(best, capped, problems)
