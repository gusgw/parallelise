"""The check: an independent judgement of a finished work folder.

The check judges the agent's routine and nothing else that the agent wrote. It never builds
or runs the agent's own test program for the verdict, and it never uses the Makefile, the
benchmark or any other trusted file from the work folder. It assembles a fresh judging folder
in a temporary directory from the problem's own files, in the way a work folder is assembled,
and copies into it two things only: the agent's routine (``<name>.c``, or ``<name>.cu`` for
CUDA) and the problem's own known-good test program (``reference/test_<name>.c``), which
compares the routine's answer with the expected values in ``reference.h``. The judging folder
passes only if it compiles with warnings treated as errors, the problem's test passes at one
worker and at several, its benchmark runs, and the benchmark's speed-up figures survive the
rules in ``bench_check``. A routine that computes the wrong answer therefore fails, however
weak or empty the agent's own test is.

For the record only, and after the verdict is settled, the check also builds the agent's own
test program against the agent's routine in a second fresh folder, runs it at one worker, and
adds one line to the log saying whether it passed. That line never changes the verdict.

Two terms need explaining.

The floor is the least best speed-up that the benchmark table must reach. A speed-up is
the time taken by one worker divided by the time taken by several, so a floor of 1.5 means
that the best row of the table must show the work finishing at least one and a half times
faster than with one worker. Each problem sets its own floor in the ``SPEEDUP_FLOOR``
setting of its Makefile. When the Makefile sets none, the floor is 1.5 for OpenMP and MPI.
Problems limited by the speed of memory rather than by arithmetic use 1.25, because extra
workers cannot read memory much faster. CUDA problems use 3. A caller may ask for a
stricter floor than the Makefile's, but not a looser one.

The tampered stage is the first stage of the check. It exists so that an agent which breaks
the rules fails, even though the verdict no longer depends on the work folder's trusted
files. It fails when a trusted file in the work folder differs from the fingerprint (a
SHA-256 digest) taken when the work folder was assembled, or from the copy in the freshly
assembled judging folder, or when the work folder holds a file named ``GNUmakefile`` or
``makefile``, which ``make`` reads in preference to ``Makefile``. The comparison with the
fresh copy is the one that cannot be defeated by rewriting the fingerprints, which are kept
in the work folder. The trusted files are the ones the agent must not change: ``bench.c``
(the benchmark program), ``Makefile``, ``testing.h`` (the helper for the unit test),
``bench_sweep.sh``, ``check_bench.py`` and ``check_bench_cuda.py`` (the small scripts that
call the table rules), ``reference.h`` (the expected values), ``problem.h`` (helpers
shared between backends, when the problem has one) and the problem's own header
``<name>.h``, which gives the signature of the routine. A missing or unreadable record of
the fingerprints also fails this stage.

When the check fails, the judging folder is kept as ``check-fail`` inside the work folder, so
that a reader can see exactly what was judged. When the check passes, it is deleted.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from parallelise.assemble import (
    BACKENDS,
    TRUSTED_MANIFEST,
    AssembleError,
    assemble_work_dir,
    file_digest,
    routine_filename,
    routine_name,
    trusted_digest,
    trusted_file_names,
    unit_test_filename,
)
from parallelise.bench_check import (
    DEFAULT_CUDA_FLOOR,
    DEFAULT_FLOOR,
    BenchVerdict,
    check_cpu_table,
    check_gpu_table,
)
from parallelise.config import without_secrets

log = logging.getLogger("parallelise.check")

#: Stages at which a run can fail; "none" means it passed.
FAILURE_STAGES = (
    "none",
    "no-files",
    "tampered",
    "build",
    "test-one-worker",
    "test-many-workers",
    "benchmark",
    "timeout",
)

#: Name of the folder, inside the work folder, in which a failed check keeps the judging folder.
KEPT_FOLDER = "check-fail"

#: The most seconds the agent's own test program may run, for the log only, at the end.
AGENT_TEST_LIMIT_S = 60

#: Files that ``make`` reads in preference to ``Makefile``; a work folder must not hold them.
SHADOWING_MAKEFILES = ("GNUmakefile", "makefile")

# OpenMP code needs -fopenmp; MPI code is built by the MPI compiler wrapper, which adds
# what it needs, and must not be given it. CUDA flags are already in the Makefile.
STRICT_CFLAGS = {
    "openmp": "-O2 -fopenmp -Wall -Wextra -Werror -std=c11",
    "mpi": "-O2 -Wall -Wextra -Werror -std=c11",
}


@dataclass
class CheckResult:
    """What the check found: whether it passed, where it failed, and the full log."""

    passed: bool
    failure_stage: str
    speedup_best: float | None
    log: str


class _Timeout(Exception):
    pass


def require_reference_test(problem_dir: Path, backend: str) -> Path:
    """Return the problem's known-good test program for ``backend``, or raise AssembleError.

    The check judges every routine with this file, so a problem without it cannot be checked.
    A caller that is about to start an expensive run can use this to find that out first.
    """
    problem_dir = Path(problem_dir)
    if backend not in BACKENDS:
        raise AssembleError(f"unknown backend {backend!r}; choose one of {', '.join(BACKENDS)}")
    name = routine_name(problem_dir)
    path = problem_dir / backend / "reference" / unit_test_filename(name)
    if not path.is_file():
        raise AssembleError(
            f"The problem {problem_dir.name} has no known-good test program {path}. The check "
            "judges a routine with that file, so add it to the problem or choose another problem."
        )
    return path


def assemble_judging_folder(
    problem_dir: Path,
    backend: str,
    dest: Path,
    routine: Path | None = None,
    test: Path | None = None,
) -> Path:
    """Build a fresh folder in which a routine can be judged, and return it.

    The folder is assembled from the problem's own files exactly as a work folder is
    (``assemble_work_dir``), so every trusted file comes from the problem and none from any
    work folder. Then two files are added, and only those: ``routine``, the routine source
    to judge (when given), and the test program, which defaults to the problem's known-good
    test, ``reference/test_<name>.c``. Pass the agent's own test file as ``test`` to build
    that instead. The test is installed under the name the Makefile expects.
    """
    problem_dir, dest = Path(problem_dir), Path(dest)
    name = routine_name(problem_dir)
    test_source = require_reference_test(problem_dir, backend) if test is None else Path(test)
    assemble_work_dir(problem_dir, backend, dest)
    if routine is not None:
        shutil.copy2(routine, dest / routine_filename(name, backend))
    shutil.copy2(test_source, dest / unit_test_filename(name))
    return dest


def _makefile_floor(folder: Path, backend: str) -> float:
    """Return the speed-up floor the Makefile asks for, or the default for the backend."""
    text = (folder / "Makefile").read_text(encoding="utf-8", errors="replace")
    match = re.search(r"SPEEDUP_FLOOR=([0-9.]+)", text)
    if match:
        return float(match.group(1))
    return DEFAULT_CUDA_FLOOR if backend == "cuda" else DEFAULT_FLOOR


def _check_recipe(folder: Path) -> dict[int, str]:
    """Return the recipe lines of the ``check`` target, keyed by line number."""
    lines = (folder / "Makefile").read_text(encoding="utf-8", errors="replace").splitlines()
    recipe: dict[int, str] = {}
    in_check = False
    for number, line in enumerate(lines, start=1):
        if re.match(r"^check\s*:", line):
            in_check = True
        elif in_check and line.startswith("\t"):
            recipe[number] = line.strip()
        elif in_check and line.strip() and not line.startswith("#"):
            break
    return recipe


def _classify_check_failure(folder: Path, output: str) -> str:
    """Decide which stage of ``make check`` failed from the line that make reports."""
    match = re.search(r"\[[^\]:]*:(\d+): check\]", output)
    recipe = _check_recipe(folder)
    if match and int(match.group(1)) in recipe:
        failing = int(match.group(1))
        command = recipe[failing]
        if "check_bench" in command or "bench" in command and "test_" not in command:
            return "benchmark"
        test_lines = [n for n, c in recipe.items() if "test_" in c and "bench" not in c]
        if failing in test_lines:
            return "test-one-worker" if failing == test_lines[0] else "test-many-workers"
    if "BENCH CHECK FAIL" in output:
        return "benchmark"
    if "FAIL:" in output:
        return "test-one-worker"
    return "benchmark"


def _digest_or_none(path: Path, digest: Callable[[Path], str] = file_digest) -> str | None:
    """Return a file's digest, or None when the file is absent or cannot be read."""
    try:
        return digest(path) if path.is_file() else None
    except OSError:
        return None


def _tampered_files(work_dir: Path, fresh_dir: Path, name: str) -> list[str]:
    """Return a description of each rule that the work folder breaks; empty if it keeps all.

    Three things are looked for. A file named ``GNUmakefile`` or ``makefile`` makes ``make``
    ignore the trusted ``Makefile``. A trusted file that differs from the digest recorded
    in ``TRUSTED_MANIFEST`` at assembly has been changed since then; a missing or unreadable
    manifest is reported too, because it was meant to be there. And a trusted file that
    differs from the copy in ``fresh_dir``, a folder just assembled from the problem's own
    files, has been changed even if the manifest was rewritten to match.
    """
    problems: list[str] = []
    for filename in SHADOWING_MAKEFILES:
        path = work_dir / filename
        if path.exists() or path.is_symlink():
            problems.append(f"{filename} is present, and make would read it instead of Makefile")

    reported: set[str] = set()
    try:
        expected = json.loads((work_dir / TRUSTED_MANIFEST).read_text(encoding="utf-8"))["sha256"]
        if not isinstance(expected, dict) or not expected:
            raise ValueError("no digests recorded")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problems.append(f"{TRUSTED_MANIFEST} is missing or unreadable ({exc})")
        expected = {}
    for filename, digest in sorted(expected.items()):
        found = _digest_or_none(work_dir / filename)
        if found is None:
            problems.append(f"{filename} has been removed")
            reported.add(filename)
        elif found != digest:
            problems.append(f"{filename} has been changed")
            reported.add(filename)

    for filename in trusted_file_names(name):
        if filename in reported or not (fresh_dir / filename).is_file():
            continue
        found = _digest_or_none(work_dir / filename, trusted_digest)
        if found is None:
            problems.append(f"{filename} has been removed")
        elif found != trusted_digest(fresh_dir / filename):
            problems.append(
                f"{filename} differs from the copy made from the problem's own files"
            )
    return problems


def _run(command: list[str], cwd: Path, env: dict[str, str], deadline: float) -> tuple[int, str]:
    """Run a command, merging its output, and kill it and its children at the deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _Timeout
    log.debug("running %s in %s", command, cwd)
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        os.killpg(proc.pid, signal.SIGKILL)
        partial = proc.communicate()[0] or ""
        raise _Timeout(partial or (exc.output or "")) from exc
    return proc.returncode, output


def _environment(backend: str) -> dict[str, str]:
    """Return the environment for the build tools and for the code under judgement.

    It is the caller's environment without anything secret (see ``without_secrets``), with
    the strict compiler flags, and with this interpreter and package found first, because
    the Makefiles run ``python3 check_bench.py``.
    """
    env = without_secrets()
    if backend in STRICT_CFLAGS:
        env["CFLAGS"] = STRICT_CFLAGS[backend]
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    package_parent = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = package_parent + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("SPEEDUP_FLOOR", None)
    return env


def _judge(
    judging: Path,
    name: str,
    backend: str,
    env: dict[str, str],
    deadline: float,
    timeout_s: int,
    speedup_floor: float | None,
    record: list[str],
) -> tuple[str, float | None]:
    """Build and test the routine in the judging folder; return the stage and best speed-up.

    The stage is "none" when everything passed. Output and reasons are appended to ``record``.
    """
    best: float | None = None
    try:
        record.append("[check] build with warnings as errors")
        code, out = _run(["make", f"test_{name}", "bench"], judging, env, deadline)
        record.append(out)
        if code != 0:
            return "build", None

        record.append("[check] the problem's test at one worker and at several, then the benchmark")
        code, out = _run(["make", "check"], judging, env, deadline)
        record.append(out)
        csv_path = judging / "bench.csv"
        floor = speedup_floor if speedup_floor is not None else _makefile_floor(judging, backend)
        verdict: BenchVerdict | None = None
        if csv_path.is_file():
            checker = check_gpu_table if backend == "cuda" else check_cpu_table
            verdict = checker(csv_path, floor)
            best = verdict.speedup_best
        if code != 0:
            return _classify_check_failure(judging, out), best
        if verdict is None:
            record.append("[check] the benchmark wrote no bench.csv")
            return "benchmark", None
        if not verdict.ok:
            record.append("[check] " + verdict.message)
            return "benchmark", best
    except _Timeout as exc:
        if exc.args and exc.args[0]:
            record.append(str(exc.args[0]))
        record.append(f"[check] time limit of {timeout_s} s reached")
        return "timeout", None
    except OSError as exc:
        record.append(f"[check] could not run the build tools: {exc}")
        return "build", None
    return "none", best


def _one_worker_command(name: str, backend: str) -> tuple[list[str], dict[str, str]]:
    """Return the command that runs a built test program with one worker.

    The second value is the environment variables that the command needs on top of the
    usual ones: one OpenMP thread, or none for the other backends.
    """
    program = f"./test_{name}"
    if backend == "openmp":
        return [program], {"OMP_NUM_THREADS": "1"}
    if backend == "mpi":
        return ["mpirun", "--oversubscribe", "-np", "1", program], {}
    return [program], {}


def _agent_test_line(
    work_dir: Path,
    problem_dir: Path,
    backend: str,
    name: str,
    env: dict[str, str],
    deadline: float,
    scratch: Path,
) -> str:
    """Build and run the agent's own test program at one worker, and describe what happened.

    This is for the log only. It is built in a second fresh folder with the trusted
    Makefile, against the agent's routine, so that nothing in the work folder is used to
    build it. Whatever happens, the verdict is not changed.
    """
    filename = unit_test_filename(name)
    prefix = f"[check] For the record only, with no effect on the verdict, the agent's {filename}"
    source = work_dir / filename
    if not source.is_file():
        return f"{prefix} does not exist, so it was not run."
    if deadline - time.monotonic() <= 0:
        return f"{prefix} was not run, because the time limit of the check had been reached."
    routine = work_dir / routine_filename(name, backend)
    try:
        folder = assemble_judging_folder(
            problem_dir, backend, scratch / "agent-test", routine, test=source
        )
        code, _ = _run(["make", f"test_{name}"], folder, env, deadline)
        if code != 0:
            return f"{prefix} did not build."
        command, extra = _one_worker_command(name, backend)
        code, _ = _run(command, folder, {**env, **extra}, deadline)
    except _Timeout:
        return f"{prefix} did not finish before the time limit."
    except (OSError, AssembleError) as exc:
        return f"{prefix} could not be run ({exc})."
    if code == 0:
        return f"{prefix} built and passed at one worker."
    return f"{prefix} built but failed at one worker (exit status {code})."


def _discard(path: Path) -> None:
    """Delete a file, a link or a folder if it exists, and never follow a link."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _keep_judging_folder(judging: Path, work_dir: Path, record: list[str]) -> None:
    """Move the judging folder into the work folder as ``check-fail`` and say so in the log."""
    target = work_dir / KEPT_FOLDER
    try:
        _discard(target)
        shutil.move(str(judging), str(target))
    except OSError as exc:
        record.append(f"[check] the judging folder could not be kept in {work_dir} ({exc})")
        return
    record.append(f"[check] the folder that was judged is kept as {target}")


def run_check(
    work_dir: Path,
    problem_dir: Path,
    backend: str,
    timeout_s: int = 300,
    speedup_floor: float | None = None,
) -> CheckResult:
    """Judge the routine in a work folder; the whole judgement shares one limit of ``timeout_s``.

    ``problem_dir`` is the problem folder the work folder was assembled from (the masked copy
    for a masked run). Everything the verdict rests on comes from there: the check assembles
    a fresh judging folder in a temporary directory, copies in only the agent's routine and
    the problem's own known-good test program, and builds and runs that. The work folder is
    first examined for broken rules, which gives the stage ``tampered`` (see the module
    description). The judging folder is deleted if the check passes and kept as ``check-fail``
    in the work folder if it fails. Raises ``AssembleError`` if the problem folder cannot
    be used, for example when it lacks the known-good test.

    The speed-up floor normally comes from the problem's Makefile. When ``speedup_floor``
    is given, the benchmark table is checked against it as well, so a stricter floor than
    the Makefile's takes effect (a looser one cannot, because the Makefile's own check
    runs first).
    """
    work_dir = Path(work_dir).resolve()
    problem_dir = Path(problem_dir).resolve()
    name = routine_name(problem_dir)
    require_reference_test(problem_dir, backend)
    record: list[str] = []
    deadline = time.monotonic() + timeout_s
    _discard(work_dir / KEPT_FOLDER)

    scratch = Path(tempfile.mkdtemp(prefix="parallelise-check-"))
    try:
        routine = work_dir / routine_filename(name, backend)
        judging = assemble_judging_folder(
            problem_dir, backend, scratch / "judge", routine if routine.is_file() else None
        )
        best: float | None = None
        changed = _tampered_files(work_dir, judging, name)
        if changed:
            record.append("[check] the work folder breaks the rules about trusted files:")
            record.extend(f"[check]   {line}" for line in changed)
            stage = "tampered"
        elif not routine.is_file():
            record.append(f"[check] missing files: {routine.name}")
            stage = "no-files"
        else:
            env = _environment(backend)
            stage, best = _judge(
                judging, name, backend, env, deadline, timeout_s, speedup_floor, record
            )
            # The agent's own test gets a short time of its own, within the overall limit.
            test_deadline = min(deadline, time.monotonic() + AGENT_TEST_LIMIT_S)
            record.append(
                _agent_test_line(work_dir, problem_dir, backend, name, env, test_deadline, scratch)
            )
        passed = stage == "none"
        if not passed:
            _keep_judging_folder(judging, work_dir, record)
        record.append(f"CHECK {'PASS' if passed else 'FAIL'} ({stage}) {work_dir}")
        return CheckResult(passed, stage, best, "\n".join(record))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
