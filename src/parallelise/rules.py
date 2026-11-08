"""The rule on what a routine may contain, checked before any routine is judged.

A routine may read no file but its own header and the system's headers, hold no assembly, define
no name but its own, call only the functions on a short list, make no system call and hold no
code that runs before the program starts; the section "The rule" of docs/DESIGN.md gives the
reasons. Each backend has its own list of calls, and a CUDA routine may hold the C++ names and
start-up code that the CUDA compiler adds itself.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .config import without_secrets
from .errors import ParalleliseError
from .generate import COMPILERS
from .routine import C_NAMES

# Each backend's own calls, as patterns, and the calls among them that are not allowed.
_BACKEND_CALLS = {
    "serial": [],
    "openmp": [r"GOMP_\w+", r"omp_\w+"],
    "mpi": [r"MPI_\w+", r"PMPI_\w+", r"ompi_mpi_\w+"],
    "cuda": [r"cuda\w+", r"__cuda\w+", r"__gxx_personality_v0", r"_Unwind_Resume", r"__cxa_\w+",
             r"_Zn[wa]m", r"_Zd[la]Pv(m)?", r"atexit"],
}
_FORBIDDEN_CALLS = re.compile(
    r"omp_get_wtime|omp_get_wtick|cudaEventElapsedTime|"
    r"(MPI|PMPI)_(Wtime|Wtick|Comm_spawn\w*|File_\w+|Open_port|Comm_connect|Comm_accept)"
)

_ASSEMBLY = re.compile(r"\b(asm|__asm|__asm__)\b|#\s*embed\b")
_STRING = re.compile(r'"(?:\\.|[^"\\\n])*"')
_MARKER = re.compile(r'#\s*(?:line\s+)?\d+\s+"([^"]*)"')
_STARTUP_SECTIONS = re.compile(r"\.(init_array|preinit_array|ctors)\b")
_SYSTEM_CALL = re.compile(r"\s(syscall|sysenter|int\s+\$0x80)\b")


def _run(command: list[str], folder: Path) -> subprocess.CompletedProcess[str]:
    """Run a tool of the build in ``folder``, with the environment that compiled code gets."""
    try:
        return subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=300,
                              env=without_secrets())  # fmt: skip
    except FileNotFoundError:
        raise ParalleliseError(
            f"The judgement needs {command[0]}, which is not installed here. Install it, or "
            "judge the routine on a machine that has it."
        ) from None


def _output(command: list[str], folder: Path) -> str:
    proc = _run(command, folder)
    if proc.returncode != 0:
        raise ParalleliseError(f"{command[0]} failed: {(proc.stderr or proc.stdout)[-500:]}")
    return proc.stdout


def _allowed_call(symbol: str, backend: str) -> bool:
    # The C library's checked forms, such as __fprintf_chk, stand for the plain ones.
    plain = re.fullmatch(r"__(\w+)_chk", symbol)
    if symbol in C_NAMES or (plain and plain.group(1) in C_NAMES):
        return True
    if _FORBIDDEN_CALLS.fullmatch(symbol):
        return False
    return any(re.fullmatch(p, symbol) for p in _BACKEND_CALLS[backend])


def _system_header(path: Path) -> bool:
    """True for a header that the system or a compiler installed, in an include folder."""
    return path.parts[:2] in (("/", "usr"), ("/", "opt")) and "include" in path.parts


def _own_code(expanded: str, source: str) -> str:
    """The lines of the preprocessor's output that come from ``source`` itself, without their
    strings."""
    own, kept = False, []
    for line in expanded.splitlines():
        marker = _MARKER.match(line)
        if marker:
            own = marker.group(1).endswith(source)
        elif own:
            kept.append(_STRING.sub('""', line))
    return "\n".join(kept)


def _own_lines(message: str, source: str) -> str:
    """The compiler's lines about the routine's own file, which quote no other file."""
    return "\n".join(line for line in message.splitlines()
                     if line.startswith((f"{source}:", f"{source}(")))  # fmt: skip


def check_routine(folder: Path, name: str, backend: str, source: str,
                  target: str = "routine.o") -> tuple[str, str] | None:
    """Check the routine in ``source`` against the rule, building its object file ``target``.

    ``folder`` holds the routine, its generated header and the Makefile. The result is None for
    an allowed routine, and otherwise the stage and the reason: "build" for a routine that does
    not compile and "rule" for one that breaks the rule. A missing tool raises ParalleliseError,
    because that is a fault of the machine and not of the routine.
    """
    cc, flags = COMPILERS[backend]
    listed = _run([cc, *flags.split(), "-M", source], folder)
    if listed.returncode != 0:
        said = _own_lines(listed.stderr, source) or "the compiler could not read it"
        return "build", f"it does not compile:\n{said}"
    words = listed.stdout.replace("\\\n", " ").split()
    # The list begins with the object file and a colon, which some compilers write apart.
    start = next(i for i, w in enumerate(words) if w.endswith(":")) + 1
    own = {(folder / f"{name}.h").resolve(), (folder / source).resolve()}
    foreign = [w for w in words[start:] if w != "\\" and (folder / w).resolve() not in own
               and not _system_header((folder / w).resolve())]  # fmt: skip
    if foreign:
        return "rule", f"it reads {', '.join(foreign)}; it may read only its own and system headers"
    expanded = _run([cc, *flags.split(), "-E", source], folder)
    if expanded.returncode != 0:
        said = _own_lines(expanded.stderr, source) or "the compiler could not read it"
        return "build", f"it does not compile:\n{said}"
    if _ASSEMBLY.search(_own_code(expanded.stdout, source)):
        return "rule", "it holds assembly language, which the rule forbids"
    built = _run(["make", "-s", target], folder)
    if built.returncode != 0:
        return "build", f"it does not compile:\n{built.stderr[-2000:]}"
    problems = []
    defined = _output(["nm", "-g", "--defined-only", target], folder).split()[2::3]
    # The CUDA compiler adds C++ names of its own; but no routine may replace C++'s operators
    # new and delete, which the whole program would then use.
    extra = [s for s in defined if s != name and not (backend == "cuda" and s.startswith("_Z")
                                                      and not re.match(r"_Z(n[wa]|d[la])", s))]
    if extra:
        problems.append(
            f"it defines {', '.join(sorted(extra))}; every function but {name} must be static"
        )
    called = _output(["nm", "-u", target], folder).split()[1::2]
    refused = sorted({s for s in called if not _allowed_call(s, backend)})
    if refused:
        problems.append(
            f"it calls {', '.join(refused)}; a routine may use only the C library's mathematics, "
            "memory and printing functions and its backend's own calls, apart from their clocks "
            "and anything that starts a process"
        )
    if _SYSTEM_CALL.search(_output(["objdump", "-d", target], folder)):
        problems.append("it makes a system call of its own")
    if backend != "cuda" and _STARTUP_SECTIONS.search(_output(["objdump", "-h", target], folder)):
        problems.append("it holds code that runs before the program starts")
    return ("rule", "; ".join(problems)) if problems else None
