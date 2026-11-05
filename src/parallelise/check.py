"""The check: an independent judgement of a finished work folder.

The check judges the agent's routine and nothing else that the agent wrote. It never builds
or runs the agent's own test program for the verdict, and it never uses the Makefile, the
benchmark or any other trusted file from the work folder. It assembles a fresh judging folder
in a temporary directory from the problem's own files, in the way a work folder is assembled,
and copies into it two things only: the agent's routine (``<name>.c``, or ``<name>.cu`` for
CUDA) and the problem's own known-good test program (``reference/test_<name>.c``), which
compares the routine's answer with the expected values in ``reference.h``. The routine passes
when it compiles with warnings treated as errors, the problem's test passes at one worker and
at several, the benchmark builds and runs, and the benchmark's table is sound by the rules in
``bench_check``: it is well formed and shows no sign of invented figures. A routine that
computes the wrong answer therefore fails, however weak or empty the agent's own test is.

For OpenMP, the check also watches the benchmark's calls to the routine, because the
benchmark alone never looks at the answers and takes its one-worker baseline from the routine
itself. It links a small trusted C file, the call recorder (see ``recorder.py``), into the
benchmark, so that each call passes through it. Two rules follow. First, the benchmark is run
a second time with the problem's own known-good routine (``reference/<name>.c``) called beside
the agent's on every call, and the routine fails at the stage ``benchmark`` if any answer
differs; so a routine that is right only at the test's small sizes fails. Second, in the timed
run each call's processor time, summed over all its threads, is divided by its wall time to
give the number of processors the call kept busy. A row of the table whose speed-up is more
than 1.2 times that number is credited with only that number, so a serial routine, which keeps
one processor busy, cannot be reported as faster, however it times itself. A row that matches
no recorded call fails the stage ``benchmark``. MPI and CUDA benchmarks are not watched.

Passing means that the routine is correct and kept the rules; it says nothing about speed.
Speed is reported separately, as one of the verdicts in ``bench_check``: "faster", "no
useful speed-up", "speed not judged on this machine" or "not measured". A correct routine
that is not faster passes with the verdict "no useful speed-up", because a parallel method
that gives no speed-up for some algorithm on some machine is a finding, not a failed
conversion. A run that fails has the speed verdict "not measured".

For the record only, and after the verdict is settled, the check also builds the agent's own
test program against the agent's routine in a second fresh folder, runs it at one worker, and
adds one line to the log saying whether it passed. That line never changes the verdict.

Two terms need explaining.

The floor is the least best speed-up that counts as a useful speed-up. A speed-up is the
time taken by one worker divided by the time taken by several, so a floor of 1.5 means that
the best row of the table must show the work finishing at least one and a half times faster
than with one worker. Each problem sets its own floor in the ``SPEEDUP_FLOOR`` setting of its
Makefile. When the Makefile sets none, the floor is 1.5 for OpenMP and MPI. Problems limited
by the speed of memory rather than by arithmetic use 1.25, because extra workers cannot read
memory much faster. CUDA problems use 3. A caller may ask for a stricter floor than the
Makefile's, but not a looser one. The floor decides the speed verdict and never the pass.

The floor is used in two places. In the agent's own work folder, ``make check`` fails when
the floor is missed, because that is how the agent learns that it has not yet parallelised
its routine. In the judging folder, the check sets ``BENCH_FLOOR_ROLE=report``, so that the
same Makefile reports a missed floor as "no useful speed-up" and carries on. A malformed or
invented table fails in both places.

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

A work folder can also be judged as assembled (``as_assembled``), for re-judging an old work
folder after the problem's trusted files have changed. The trusted files then come from the
work folder itself, each checked against the folder's own fingerprints, rather than from the
problem; the routine, the problem's current known-good test and known-good routine, and the
present rules are used as usual. Since the fingerprints are kept in the work folder, this mode
is weaker than the ordinary check, and it is meant only for folders whose ordinary check
reports ``tampered`` because the problem has moved on.
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
from dataclasses import dataclass, replace
from pathlib import Path

from parallelise import recorder
from parallelise.assemble import (
    BACKENDS,
    CHECKER_SCRIPTS,
    TRUSTED_MANIFEST,
    AssembleError,
    assemble_work_dir,
    file_digest,
    routine_filename,
    routine_name,
    trusted_digest,
    trusted_file_names,
    unit_test_filename,
    write_checker_scripts,
)
from parallelise.bench_check import (
    DEFAULT_CUDA_FLOOR,
    DEFAULT_FLOOR,
    FASTER,
    FLOOR_ROLE_VARIABLE,
    JUDGE_SPEED_VARIABLE,
    NO_USEFUL_SPEEDUP,
    NOT_MEASURED,
    REPORT_ROLE,
    SPEED_NOT_JUDGED,
    BenchVerdict,
    check_cpu_table,
    check_gpu_table,
    sweep_rows,
)
from parallelise.config import without_secrets
from parallelise.errors import CheckError

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

#: Variables removed from the environment of the code under judgement. The two OpenMP ones
#: decide whether idle threads keep a processor busy while they wait; left at their defaults,
#: idle threads soon stop, so the processor time of a call reflects real work.
UNSET_VARIABLES = (
    "OMP_WAIT_POLICY", "GOMP_SPINCOUNT", recorder.TIMES_VARIABLE, recorder.VERIFY_VARIABLE,
)  # fmt: skip

# OpenMP code needs -fopenmp; MPI code is built by the MPI compiler wrapper, which adds
# what it needs, and must not be given it. CUDA flags are already in the Makefile.
STRICT_CFLAGS = {
    "openmp": "-O2 -fopenmp -Wall -Wextra -Werror -std=c11",
    "mpi": "-O2 -Wall -Wextra -Werror -std=c11",
}


@dataclass
class CheckResult:
    """What the check found: whether the routine is correct, how fast it is, and the full log.

    ``passed`` is true when the routine is correct and kept the rules, whatever its speed, and
    ``failure_stage`` names the stage at which it failed ("none" when it passed). ``speed_verdict``
    is one of the verdicts in ``bench_check``, and is "not measured" for a run that failed.
    ``speedup_best`` is the best speed-up in the benchmark table, when there was one, and
    ``speedup_floor`` the floor the speed verdict was judged against.
    """

    passed: bool
    failure_stage: str
    speedup_best: float | None
    log: str
    speed_verdict: str = NOT_MEASURED
    speedup_floor: float | None = None


class _Timeout(Exception):
    pass


def require_reference_test(problem_dir: Path, backend: str) -> Path:
    """Return the problem's known-good test program for ``backend``, or raise AssembleError.

    The check judges every routine with this file, so a problem without it cannot be checked.
    A caller that is about to start an expensive run can use this to find that out first.
    """
    problem_dir = Path(problem_dir)
    if backend not in BACKENDS:
        raise AssembleError(
            f"The backend {backend!r} is not one of {', '.join(BACKENDS)}. Choose one of them."
        )
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


def _floor(folder: Path, backend: str, requested: float | None) -> float:
    """Return the floor to judge against: the Makefile's, or a stricter one that was asked for."""
    floor = _makefile_floor(folder, backend)
    return floor if requested is None else max(floor, requested)


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


def speed_judged() -> bool:
    """Return False when PARALLELISE_JUDGE_SPEED is 0, so that only correctness is judged.

    Shared test servers, such as the ones continuous integration runs on, have too few and
    too busy processor cores for their timings to mean anything. There the check still
    builds, tests and runs the benchmark, so it proves that the code works, but it does not
    judge the speed-up.
    """
    return os.environ.get("PARALLELISE_JUDGE_SPEED", "1").strip() != "0"


def _environment(backend: str) -> dict[str, str]:
    """Return the environment for the build tools and for the code under judgement.

    It is the caller's environment without anything secret (see ``without_secrets``), with
    the strict compiler flags, and with this interpreter and package found first, because
    the Makefiles run ``python3 check_bench.py``. ``BENCH_FLOOR_ROLE`` is set to ``report``,
    so that the benchmark script reports a missed floor instead of failing, and
    ``BENCH_JUDGE_SPEED`` is set to 0 when speed is not judged on this machine. The
    variables in ``UNSET_VARIABLES`` are removed, so that the OpenMP library waits in its
    usual way and the call recorder starts switched off.
    """
    env = without_secrets()
    if backend in STRICT_CFLAGS:
        env["CFLAGS"] = STRICT_CFLAGS[backend]
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    package_parent = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = package_parent + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("SPEEDUP_FLOOR", None)
    env.pop(JUDGE_SPEED_VARIABLE, None)
    for name in UNSET_VARIABLES:
        env.pop(name, None)
    env[FLOOR_ROLE_VARIABLE] = REPORT_ROLE
    if not speed_judged():
        env[JUDGE_SPEED_VARIABLE] = "0"
    return env


def _judge(
    judging: Path,
    name: str,
    backend: str,
    env: dict[str, str],
    deadline: float,
    timeout_s: int,
    floor: float,
    record: list[str],
    known_good: Path | None = None,
) -> tuple[str, BenchVerdict | None]:
    """Build and test the routine in the judging folder; return the stage and the table's verdict.

    The stage is "none" when the routine is correct and its benchmark table is sound, whatever
    its speed. The verdict is None when no benchmark table was written. Output and reasons are
    appended to ``record``. For OpenMP, ``known_good`` is the problem's known-good routine:
    the benchmark is then built with the call recorder (see ``recorder.py``), its answers are
    compared with the known-good routine's at the benchmark's own size, and each row's
    speed-up is credited only as far as the processors its call kept busy allow.
    """
    verdict: BenchVerdict | None = None
    try:
        record.append("[check] build with warnings as errors")
        code, out = _run(["make", f"test_{name}"], judging, env, deadline)
        record.append(out)
        if code != 0:
            return "build", None
        link = None
        if backend == "openmp" and known_good is not None:
            link = _prepare_recorder(judging, known_good, env, deadline, record)
        bench = ["make", "bench"] + ([f"LDLIBS={link}"] if link else [])
        code, out = _run(bench, judging, env, deadline)
        record.append(out)
        if code != 0:
            return "build", None

        record.append("[check] the problem's test at one worker and at several, then the benchmark")
        check_env = dict(env)
        if link:
            check_env[recorder.TIMES_VARIABLE] = str(judging / recorder.TIMES_FILE)
        code, out = _run(["make", "check"], judging, check_env, deadline)
        record.append(out)
        csv_path = judging / "bench.csv"
        if csv_path.is_file():
            checker = check_gpu_table if backend == "cuda" else check_cpu_table
            verdict = checker(csv_path, floor, judge_speed=speed_judged())
        if code != 0:
            return _classify_check_failure(judging, out), verdict
        if verdict is None:
            record.append("[check] the benchmark wrote no bench.csv")
            return "benchmark", None
        if not verdict.sound:
            record.append("[check] " + verdict.message)
            return "benchmark", verdict
        if link:
            if not _answers_agree(judging, env, deadline, record):
                return "benchmark", verdict
            verdict, supported = _credit_speed(judging, verdict, floor, record)
            if not supported:
                return "benchmark", verdict
    except _Timeout as exc:
        if exc.args and exc.args[0]:
            record.append(str(exc.args[0]))
        record.append(f"[check] time limit of {timeout_s} s reached")
        return "timeout", None
    except OSError as exc:
        record.append(f"[check] could not run the build tools: {exc}")
        return "build", None
    return "none", verdict


def _prepare_recorder(
    judging: Path, known_good: Path, env: dict[str, str], deadline: float, record: list[str]
) -> str | None:
    """Write and compile the call recorder and the known-good routine in the judging folder.

    Returns the value of ``LDLIBS`` that links them into the benchmark, or None, with a line
    in the log, when they cannot be made. Both are made only from trusted files: the
    problem's header in the judging folder and the problem's own known-good routine. The
    known-good routine is compiled under another name, and every other name it defines is
    hidden, so that it cannot clash with the agent's routine.
    """
    headers = [h for h in judging.glob("*.h")
               if h.name not in ("reference.h", "problem.h", "testing.h")]  # fmt: skip
    try:
        if len(headers) != 1:
            raise recorder.PrototypeError(
                f"the judging folder holds {len(headers)} routine headers"
            )
        prototype = recorder.parse_prototype(headers[0].read_text(encoding="utf-8"))
    except (OSError, recorder.PrototypeError) as exc:
        record.append(
            f"[check] The calls to the routine cannot be watched ({exc}), so its answers at the "
            "benchmark's size and the processor time of its calls are not checked."
        )
        return None
    symbol = recorder.known_good_symbol(prototype.name)
    (judging / recorder.SOURCE_NAME).write_text(
        recorder.recorder_source(prototype, headers[0].name), encoding="utf-8"
    )
    shutil.copy2(known_good, judging / recorder.KNOWN_GOOD_SOURCE)
    cc = env.get("CC") or "gcc"
    flags = env.get("CFLAGS", STRICT_CFLAGS["openmp"]).split()
    steps = (
        [cc, *flags, "-c", recorder.SOURCE_NAME, "-o", recorder.OBJECT_NAME],
        [cc, *flags, f"-D{prototype.name}={symbol}", "-c", recorder.KNOWN_GOOD_SOURCE,
         "-o", recorder.KNOWN_GOOD_OBJECT],
        ["objcopy", f"--keep-global-symbol={symbol}", recorder.KNOWN_GOOD_OBJECT],
    )  # fmt: skip
    for command in steps:
        try:
            code, out = _run(command, judging, env, deadline)
        except OSError as exc:
            code, out = 1, str(exc)
        if code != 0:
            record.append(out)
            record.append(
                f"[check] The call recorder could not be built (the command {command[0]} "
                "failed), so the routine's answers at the benchmark's size and the processor "
                "time of its calls are not checked."
            )
            return None
    return (
        f"{recorder.OBJECT_NAME} {recorder.KNOWN_GOOD_OBJECT} -Wl,--wrap={prototype.name} -lm"
    )


def _bench_command(judging: Path, env: dict[str, str], deadline: float) -> str | None:
    """Return the shell command with which the Makefile's check target runs the benchmark.

    ``make -n`` prints the commands with every Makefile variable filled in, so this works for
    every Makefile, whatever it calls its variables. The command is the one whose output goes
    to ``bench.csv``; it comes from the trusted Makefile.
    """
    code, out = _run(["make", "-n", "check"], judging, env, deadline)
    if code != 0:
        return None
    for line in out.splitlines():
        if re.search(r">\s*bench\.csv\b", line):
            return line.strip()
    return None


def _answers_agree(
    judging: Path, env: dict[str, str], deadline: float, record: list[str]
) -> bool:
    """Run the benchmark once more, comparing each answer with the known-good routine's.

    The run is separate from the timed one, because the comparison takes time. It returns
    True when every call's answer agrees, and otherwise adds the reasons to the log.
    """
    record.append("[check] the benchmark again, comparing each answer with the known-good routine")
    command = _bench_command(judging, env, deadline)
    if command is None:
        record.append("[check] the Makefile's check target has no command that writes bench.csv")
        return False
    command = command.replace("bench.csv", recorder.VERIFY_TABLE)
    calls_file = judging / recorder.VERIFY_FILE
    code, out = _run(
        ["sh", "-c", command], judging, {**env, recorder.VERIFY_VARIABLE: str(calls_file)},
        deadline,
    )  # fmt: skip
    if out.strip():
        record.append(out)
    if code != 0:
        record.append(
            f"[check] the benchmark failed when its answers were compared (exit status {code})"
        )
        return False
    try:
        calls = recorder.read_calls(calls_file)
    except (OSError, ValueError) as exc:
        record.append(f"[check] the comparison of the answers could not be read ({exc})")
        return False
    problems = recorder.verification_problems(calls)
    if problems:
        record.append(
            "[check] At the benchmark's own size, the routine's answer is not the known-good "
            "routine's:"
        )
        record.extend(f"[check]   {line}" for line in problems)
        return False
    largest = max((c.difference / c.scale for c in calls if c.scale > 0), default=0.0)
    how = (
        f"the largest difference is {largest:.1e} of the answer's size" if largest > 0
        else "they are identical"
    )  # fmt: skip
    record.append(
        f"[check] At the benchmark's own size, all {len(calls)} answers agree with the known-good "
        f"routine's ({how})."
    )
    return True


def _credit_speed(
    judging: Path, verdict: BenchVerdict, floor: float, record: list[str]
) -> tuple[BenchVerdict, bool]:
    """Check the table against the timed calls, and return the verdict and whether it stands.

    When speed is judged, each row of the table must match a call that the recorder timed,
    and the speed-up a row is credited with is limited by the processors its call kept busy.
    The returned verdict carries the credited best speed-up and the speed verdict that follows
    from it. The second value is False when the table does not match the recorded calls.
    """
    if verdict.speed_verdict not in (FASTER, NO_USEFUL_SPEEDUP):
        return verdict, True
    try:
        calls = recorder.read_calls(judging / recorder.TIMES_FILE)
    except (OSError, ValueError) as exc:
        record.append(f"[check] the timed calls to the routine could not be read ({exc})")
        return verdict, False
    credit = recorder.credit_speedups(sweep_rows(judging / "bench.csv"), calls)
    if credit.problems or credit.best is None:
        record.append("[check] The benchmark table does not match the routine's timed calls:")
        record.extend(f"[check]   {line}" for line in credit.problems or ["no row matched"])
        return verdict, False
    record.extend(f"[check] {line[0].upper()}{line[1:]}." for line in credit.capped)
    if not credit.capped:
        return verdict, True
    best = min(credit.best, verdict.speedup_best or credit.best)
    reached = best >= floor
    reasons = [r for r in verdict.reasons if not r.startswith("no real parallel speedup")]
    if not reached:
        reasons.append(
            f"no real parallel speedup (the best speed-up that the busy processors support is "
            f"{best:.2f}, below {floor:g})"
        )
    credited = replace(
        verdict, speedup_best=best, floor_reached=reached, ok=reached, reasons=reasons,
        speed_verdict=FASTER if reached else NO_USEFUL_SPEEDUP,
    )  # fmt: skip
    return credited, True


def _speed_line(stage: str, verdict: BenchVerdict | None, floor: float) -> tuple[str, str]:
    """Return the speed verdict of a finished judgement and the log line that explains it."""
    if stage != "none" or verdict is None:
        why = (
            "the benchmark did not run, gave a wrong answer, or its table was missing, "
            "malformed, invented or not borne out by the timed calls"
            if stage == "benchmark"
            else f"the check failed at the stage {stage}"
        )  # fmt: skip
        return NOT_MEASURED, f"[check] Speed verdict: {NOT_MEASURED}, because {why}."
    speed = verdict.speed_verdict
    best = verdict.speedup_best
    shown = "unknown" if best is None else f"{best:.2f}"
    if speed == FASTER:
        why = f"the best speed-up, {shown}, reached the floor of {floor:g}"
    elif speed == NO_USEFUL_SPEEDUP:
        why = (
            f"the best speed-up, {shown}, is below the floor of {floor:g}; the routine is "
            "correct, so the check still passes"
        )
    elif speed == SPEED_NOT_JUDGED:
        why = (
            "PARALLELISE_JUDGE_SPEED is 0; the benchmark was run only to show that it works, "
            f"and its best speed-up was {shown}"
        )
    else:
        why = "the benchmark table held no speed-up to judge"
    return speed, f"[check] Speed verdict: {speed}, because {why}."


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
    as_assembled: bool = False,
) -> CheckResult:
    """Judge the routine in a work folder; the whole judgement shares one limit of ``timeout_s``.

    ``problem_dir`` is the problem folder the work folder was assembled from (the masked copy
    for a masked run). Everything the verdict rests on comes from there: the check assembles
    a fresh judging folder in a temporary directory, copies in only the agent's routine and
    the problem's own known-good test program, and builds and runs that. The work folder is
    first examined for broken rules, which gives the stage ``tampered`` (see the module
    description). The judging folder is deleted if the check passes and kept as ``check-fail``
    in the work folder if it fails. Raises ``AssembleError`` if the problem folder cannot
    be used, for example when it lacks the known-good test, and ``CheckError`` if the check
    cannot run at all, as when the computer will not let it make a temporary folder. A
    routine that fails the check is a result, not an error.

    The routine passes when it is correct and its benchmark table is sound, whatever its
    speed; the result's ``speed_verdict`` says how fast it was, and the log says why in one
    line. The speed-up floor normally comes from the problem's Makefile. When
    ``speedup_floor`` is given and is stricter than the Makefile's, the speed verdict is
    judged against it instead; a looser one has no effect.

    With ``as_assembled``, the work folder is judged as it was assembled: the trusted files
    come from the work folder itself, after each has been checked against the folder's own
    fingerprint file, instead of from the problem's current files. This is for re-judging an
    old work folder after the problem's files have changed. Only the routine, the problem's
    current known-good test program and known-good routine, and the present rules are new. A
    folder without a fingerprint file, or one that holds ``GNUmakefile`` or ``makefile``, is
    refused with ``CheckError``.
    """
    try:
        return _run_check(
            Path(work_dir), Path(problem_dir), backend, timeout_s, speedup_floor, as_assembled
        )
    except OSError as exc:
        raise CheckError(
            f"The check could not run in {work_dir} ({exc.strerror or exc}). Free the disk "
            "space or fix the permissions, then judge the work folder again with "
            "'parallelise check'."
        ) from exc


def _read_manifest(work_dir: Path) -> dict[str, str]:
    """Return the digests recorded in the work folder's fingerprint file, or raise CheckError."""
    path = work_dir / TRUSTED_MANIFEST
    if not path.is_file():
        raise CheckError(
            f"The work folder {work_dir} has no fingerprint file ({TRUSTED_MANIFEST}), so it "
            "cannot be judged as assembled. Judge it without --as-assembled."
        )
    try:
        digests = json.loads(path.read_text(encoding="utf-8"))["sha256"]
        if not isinstance(digests, dict) or not digests:
            raise ValueError("no digests are recorded")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in digests.items()):
            raise ValueError("a digest is not text")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CheckError(
            f"The fingerprint file {path} cannot be read ({exc}), so the work folder cannot be "
            "judged as assembled. Judge it without --as-assembled."
        ) from exc
    for filename in digests:
        if Path(filename).name != filename:
            raise CheckError(
                f"The fingerprint file {path} names {filename!r}, which is not a plain file "
                "name, so the work folder cannot be judged as assembled."
            )
    return digests


def _refuse_for_as_assembled(work_dir: Path, name: str) -> dict[str, str]:
    """Check that a work folder can be judged as assembled, and return its recorded digests."""
    for filename in SHADOWING_MAKEFILES:
        path = work_dir / filename
        if path.exists() or path.is_symlink():
            raise CheckError(
                f"The work folder {work_dir} holds {filename}, which make would read instead of "
                "the Makefile, so it cannot be judged as assembled."
            )
    digests = _read_manifest(work_dir)
    needed = [f for f in ("bench.c", "Makefile", f"{name}.h") if f not in digests]
    if needed:
        raise CheckError(
            f"The fingerprint file of {work_dir} does not list {', '.join(needed)}, so the "
            "folder cannot be judged as assembled. Check that --problem names the problem the "
            "folder was built from; for a masked run, give a masked copy made with the same "
            "alias."
        )
    return digests


def _manifest_problems(work_dir: Path, digests: dict[str, str]) -> list[str]:
    """Return a description of each recorded trusted file that is missing or has changed."""
    problems: list[str] = []
    for filename, digest in sorted(digests.items()):
        found = _digest_or_none(work_dir / filename)
        if found is None:
            problems.append(f"{filename} has been removed")
        elif found != digest:
            problems.append(f"{filename} has been changed")
    return problems


def assemble_as_assembled_folder(
    work_dir: Path,
    problem_dir: Path,
    backend: str,
    dest: Path,
    digests: dict[str, str],
    routine: Path | None,
) -> Path:
    """Build a judging folder from a work folder's own trusted files, and return it.

    Every file named in ``digests`` is copied from ``work_dir``, except the two checker
    scripts, which are written afresh so that the present table rules apply. Then the routine
    (when given) and the problem's current known-good test program are added. The caller must
    first have checked the files against ``digests``.
    """
    name = routine_name(problem_dir)
    test_source = require_reference_test(problem_dir, backend)
    dest.mkdir(parents=True)
    for filename in digests:
        if filename not in CHECKER_SCRIPTS:
            shutil.copy2(work_dir / filename, dest / filename)
    write_checker_scripts(dest)
    if routine is not None:
        shutil.copy2(routine, dest / routine_filename(name, backend))
    shutil.copy2(test_source, dest / unit_test_filename(name))
    return dest


def _differences_from_current(
    work_dir: Path, current: Path, digests: dict[str, str]
) -> list[str]:
    """Return a line for each trusted file of the work folder that the problem has since changed."""
    lines = []
    for filename in sorted(digests):
        if filename in CHECKER_SCRIPTS:
            continue
        theirs = _digest_or_none(current / filename, trusted_digest)
        if theirs is None:
            lines.append(f"{filename} is no longer one of the problem's files")
        elif theirs != _digest_or_none(work_dir / filename, trusted_digest):
            lines.append(f"{filename} differs from the problem's current copy")
    return lines


def _run_check(
    work_dir: Path,
    problem_dir: Path,
    backend: str,
    timeout_s: int,
    speedup_floor: float | None,
    as_assembled: bool = False,
) -> CheckResult:
    """Do the work of ``run_check``; an operating-system error escapes to its caller."""
    work_dir = work_dir.resolve()
    problem_dir = problem_dir.resolve()
    name = routine_name(problem_dir)
    require_reference_test(problem_dir, backend)
    digests = _refuse_for_as_assembled(work_dir, name) if as_assembled else {}
    record: list[str] = []
    deadline = time.monotonic() + timeout_s
    _discard(work_dir / KEPT_FOLDER)

    scratch = Path(tempfile.mkdtemp(prefix="parallelise-check-"))
    try:
        routine = work_dir / routine_filename(name, backend)
        present = routine if routine.is_file() else None
        if as_assembled:
            record.append(
                "[check] Judged as assembled: the trusted files come from the work folder, "
                f"checked against its own fingerprint file ({TRUSTED_MANIFEST}), and not from "
                "the problem's current files. The test program, the known-good routine and the "
                "rules are the present ones."
            )
            changed = _manifest_problems(work_dir, digests)
            if not changed:
                judging = assemble_as_assembled_folder(
                    work_dir, problem_dir, backend, scratch / "judge", digests, present
                )
                current = assemble_work_dir(problem_dir, backend, scratch / "current")
                record.extend(
                    f"[check]   {line}"
                    for line in _differences_from_current(work_dir, current, digests)
                )
            else:
                judging = scratch / "judge"
                judging.mkdir()
        else:
            judging = assemble_judging_folder(problem_dir, backend, scratch / "judge", present)
            changed = _tampered_files(work_dir, judging, name)
        floor = (
            _floor(judging, backend, speedup_floor)
            if (judging / "Makefile").is_file()
            else DEFAULT_CUDA_FLOOR if backend == "cuda" else DEFAULT_FLOOR
        )  # fmt: skip
        verdict: BenchVerdict | None = None
        if changed:
            record.append("[check] the work folder breaks the rules about trusted files:")
            record.extend(f"[check]   {line}" for line in changed)
            stage = "tampered"
        elif not routine.is_file():
            record.append(f"[check] missing files: {routine.name}")
            stage = "no-files"
        else:
            env = _environment(backend)
            known_good = problem_dir / backend / "reference" / routine_filename(name, backend)
            stage, verdict = _judge(
                judging, name, backend, env, deadline, timeout_s, floor, record,
                known_good if known_good.is_file() else None,
            )  # fmt: skip
            # The agent's own test gets a short time of its own, within the overall limit.
            test_deadline = min(deadline, time.monotonic() + AGENT_TEST_LIMIT_S)
            record.append(
                _agent_test_line(work_dir, problem_dir, backend, name, env, test_deadline, scratch)
            )
        passed = stage == "none"
        speed, line = _speed_line(stage, verdict, floor)
        record.append(line)
        if not passed:
            _keep_judging_folder(judging, work_dir, record)
        if as_assembled:
            record.append("[check] This work folder was judged as assembled.")
        record.append(f"CHECK {'PASS' if passed else 'FAIL'} ({stage}) {work_dir}")
        best = verdict.speedup_best if verdict is not None else None
        return CheckResult(passed, stage, best, "\n".join(record), speed, floor)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
