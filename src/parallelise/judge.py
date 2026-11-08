"""The judgement of a routine: whether it is correct, and how much faster than the serial routine.

The judgement is made by fixed rules, after the agent has stopped, in a fresh folder that the
agent never saw. The routine must first pass the rule on what it may contain (see ``rules``).
Then:

- Correctness at the test sizes. Fresh seeds and sizes are drawn: the declared test sizes and
  two more at random between the smallest and the largest. Each input runs in a fresh process
  at one worker and at all of them (once for CUDA), and only afterwards are the trusted answers
  computed from the notebook's Python function.
- Correctness and speed at the benchmark size. Six new inputs are drawn, one at a time. The
  routine and the serial routine each run in a process of their own, and the harness
  alternates between them: one untimed call each, then five timed calls each, on the same
  inputs. Every answer of the routine is compared with the serial routine's, and each input
  is deleted with its results once they have been compared.
- Speed. The routine and the serial routine run on the same five inputs, so they are
  compared input by input, and inputs that differ in their amount of work cannot hide a gain.
  The speed-up is the median over the inputs of the serial routine's time divided by the
  routine's, and the routine counts as faster when it beats the serial routine on all five,
  which chance alone would give once in 32 trials. Its scaling is the median ratio of its own
  time at one worker to its time at all of them, and it is called parallel when all its workers
  beat one on all five inputs. A CUDA routine has no worker count; it is called parallel when
  it launches a kernel.

Each call is timed from outside the routine's process (see ``generate``). The verdict's stage
is "none" for a correct routine and otherwise the first that failed: "rule", "build", "test",
"benchmark" or "timeout". The stage "notebook" says that the notebook's own Python failed
during the judgement, so that nothing was learnt about the routine.
"""

from __future__ import annotations

import math
import os
import random
import secrets
import select
import shutil
import signal
import statistics
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import generate, notebook, provenance, rules
from .config import without_secrets
from .routine import Declaration, compare, extents, read_results

TIMED_CALLS = 5
#: The most seconds that one call at a test size may take. An honest routine takes far less,
#: so a call that takes longer has hung, as an MPI routine can when some processes get no work.
TEST_CALL_S = 120.0
#: The memory that the inputs and results of all the MPI processes together may take.
MPI_MEMORY_BYTES = 6 * 2**30


@dataclass
class Verdict:
    """What the judgement found: correctness, and, for a correct routine, its speed."""

    correct: bool
    stage: str
    speedup: float | None = None
    faster: bool = False
    scaling: float | None = None
    parallel: bool | None = None
    log: list[str] = field(default_factory=list)


class _Stop(Exception):
    """A step of the judgement failed; the message says why, and ``stage`` where."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


class Driver:
    """A running driver: one routine in one process (or one group of MPI processes).

    ``stage`` is the stage of the judgement that a failure of the program is reported at.
    """

    def __init__(self, folder: Path, program: str, backend: str, workers: int, deadline: float,
                 stage: str = "benchmark"):
        self.deadline, self.stage = deadline, stage
        self.pipe = folder / f"marks-{program}-{workers}-{secrets.token_hex(4)}"
        os.mkfifo(self.pipe)
        # Opening the pipe for reading and writing never blocks, and keeps it open until the
        # harness closes it, whatever the driver does.
        self.marks = os.open(self.pipe, os.O_RDWR | os.O_NONBLOCK)
        self.buffer = b""
        # The caller's own OpenMP settings, such as a waiting policy that keeps idle threads
        # busy, must not reach the routine; only the number of threads is set.
        env = {k: v for k, v in without_secrets().items() if not k.startswith(("OMP_", "GOMP_"))}
        env["OMP_NUM_THREADS"] = str(workers)
        command = [f"./{program}", str(self.pipe)]
        if backend == "mpi" and program == "driver":
            command = ["mpirun", "--oversubscribe", "-np", str(workers), *command]
        # What the routine prints goes to a file, so that no amount of it can stall the driver.
        self.errors = self.pipe.with_suffix(".txt")
        with open(self.errors, "w") as errors:
            self.proc = subprocess.Popen(
                command, cwd=folder, env=env, stdin=subprocess.PIPE, stdout=errors,
                stderr=errors, text=True, start_new_session=True,
            )  # fmt: skip

    def _mark(self) -> tuple[str, float]:
        while b"\n" not in self.buffer:
            left = self.deadline - time.monotonic()
            if left <= 0:
                raise _Stop("timeout", "the time allowed for the judgement ran out")
            ready, _, _ = select.select([self.marks], [], [], min(left, 1.0))
            if ready:
                self.buffer += os.read(self.marks, 4096)
            elif self.proc.poll() is not None:
                said = self.errors.read_text(errors="replace").strip()[-300:]
                raise _Stop(self.stage, f"the program stopped ({said or self.proc.returncode})")
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.decode(), time.perf_counter()

    def call(self, inputs: Path, results: Path, timed: bool) -> float | None:
        """Make one call; return its time in seconds when it is timed."""
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(f"{'TIME' if timed else 'CALL'}\n{inputs}\n{results}\n")
            self.proc.stdin.flush()
        except BrokenPipeError:
            raise _Stop(self.stage, "the program stopped before it was called") from None
        marks = {}
        while "DONE" not in marks:
            name, when = self._mark()
            marks[name] = when
            if name == "START":  # noted; now the driver may call the routine
                try:
                    self.proc.stdin.write("GO\n")
                    self.proc.stdin.flush()
                except BrokenPipeError:
                    raise _Stop(self.stage, "the program stopped at the start of a call") from None
        return marks["END"] - marks["START"] if timed else None

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except (subprocess.TimeoutExpired, OSError):
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            self.proc.wait()
        os.close(self.marks)
        self.pipe.unlink(missing_ok=True)


def missing_tools(backend: str) -> list[str]:
    """The programs that judging a routine for ``backend`` needs and this machine lacks."""
    needed = {"make", "nm", "objdump", "gcc", generate.COMPILERS[backend][0]}
    if backend == "mpi":
        needed.add("mpirun")
    return sorted(tool for tool in needed if shutil.which(tool) is None)


def _widths(backend: str, per_process_bytes: int) -> tuple[int, ...]:
    """The worker counts to test at: one and all, or once for CUDA and for serial code."""
    if backend in ("cuda", "serial"):
        return (1,)
    most = provenance.cores()
    if backend == "mpi":
        most = max(1, min(most, MPI_MEMORY_BYTES // max(per_process_bytes, 1)))
    return (1, most) if most > 1 else (1,)


def _bytes_of(decl: Declaration, sizes: dict[str, int]) -> int:
    """The memory that one call's arguments and results take."""
    return sum(math.prod(extents(v, sizes)) * (8 if v.dtype.endswith("64") else 4)
               for v in (*decl.arguments, *decl.results))  # fmt: skip


def _test_sizes(decl: Declaration) -> list[int]:
    low, high = min(decl.test_sizes), max(decl.test_sizes)
    draw = random.SystemRandom()
    extra = [round(math.exp(draw.uniform(math.log(low), math.log(high)))) for _ in range(2)]
    return [*decl.test_sizes, *extra]


def _prepare(decl: Declaration, name: str, backend: str, routine: Path | None,
             serial: Path | None) -> Path:
    """Make a fresh folder with the generated files, the routine and the serial routine."""
    folder = Path(tempfile.mkdtemp(prefix="parallelise-judge-"))
    generate.write_build_files(folder, decl, name, backend if backend != "serial" else "openmp")
    if routine is not None:
        shutil.copy(routine, folder / generate.routine_file(name, backend))
    if serial is not None:
        shutil.copy(serial, folder / "serial.c")
    return folder


def _left(deadline: float) -> float:
    """The seconds left before ``deadline``, for a step that has a time limit of its own."""
    left = deadline - time.monotonic()
    if left <= 0:
        raise _Stop("timeout", "the time allowed for the judgement ran out")
    return left


def _build(folder: Path, name: str, backend: str, program: str) -> None:
    """Check the rule on the program's routine, then build the program; raise _Stop on failure.

    The serial routine is held to the rule for plain C, whatever the backend being judged.
    """
    if program == "serial_driver":
        source, target, backend = "serial.c", "serial.o", "serial"
    else:
        source, target = generate.routine_file(name, backend), "routine.o"
    refused = rules.check_routine(folder, name, backend, source, target)
    if refused:
        stage, why = refused
        raise _Stop(stage, ("the routine is refused: " if stage == "rule" else "") + why)
    built = subprocess.run(["make", "-s", program], cwd=folder, capture_output=True, text=True,
                           timeout=300, env=without_secrets())  # fmt: skip
    if built.returncode != 0:
        raise _Stop("build", "the program does not build:\n" + built.stderr[-2000:])


def _against_python(nb: Path, decl: Declaration, folder: Path, program: str, backend: str,
                    sizes: list[int], deadline: float) -> str:
    """Run the program on fresh inputs of each size, then compare it with the Python.

    Each input runs in a fresh process at one worker and at all of them; the trusted answers
    are computed only after every run. Raises _Stop at the first wrong answer, and returns a
    line for the log.
    """
    calls = [(size, secrets.randbits(32), folder / f"test{i}.in") for i, size in enumerate(sizes)]
    sizes_of_calls = notebook.write_input_files(nb, calls, _left(deadline))
    widths = _widths(backend, max(_bytes_of(decl, s) for s in sizes_of_calls))
    for size, _, inputs in calls:
        for workers in widths:
            limit = min(deadline, time.monotonic() + TEST_CALL_S)
            driver = Driver(folder, program, backend, workers, limit, stage="test")
            try:
                driver.call(inputs, inputs.with_suffix(f".{workers}"), timed=False)
            except _Stop as stop:
                if stop.stage == "timeout" and limit < deadline:
                    raise _Stop("timeout", f"a call at size {size} with {workers} workers took "
                                f"more than {TEST_CALL_S:g} seconds") from None
                raise
            finally:
                driver.close()
    notebook.write_answer_files(nb, [(c[2], c[2].with_suffix(".want")) for c in calls],
                                _left(deadline))
    for (size, _, inputs), call_sizes in zip(calls, sizes_of_calls, strict=True):
        want = read_results(inputs.with_suffix(".want"), decl, call_sizes)
        for workers in widths:
            wrong = compare(decl, read_results(inputs.with_suffix(f".{workers}"), decl,
                                               call_sizes), want)  # fmt: skip
            if wrong:
                raise _Stop("test", f"wrong at size {size} with {workers} workers: "
                            + "; ".join(wrong))  # fmt: skip
    return f"correct at sizes {sizes} with {list(widths)} workers"


def _failed(stop: Exception, log: list[str]) -> Verdict:
    if isinstance(stop, _Stop):
        log.append(str(stop))
        return Verdict(False, stop.stage, log=log)
    if isinstance(stop, notebook.NotebookError):
        log.append(f"the notebook failed during the judgement: {stop}")
        return Verdict(False, "notebook", log=log)
    log.append("a build took too long")
    return Verdict(False, "timeout", log=log)


def _in_fresh_folder(decl: Declaration, name: str, backend: str, routine: Path | None,
                     serial: Path | None, timeout_s: float,
                     work: Callable[[Path, float, list[str]], Verdict]) -> Verdict:
    """Run ``work`` in a fresh folder with a deadline and a log, turn a failed step into a
    verdict, and delete the folder afterwards."""
    deadline = time.monotonic() + timeout_s
    log: list[str] = []
    folder = _prepare(decl, name, backend, routine, serial)
    try:
        verdict = work(folder, deadline, log)
        verdict.log = log
        return verdict
    except (_Stop, notebook.NotebookError, subprocess.TimeoutExpired) as stop:
        return _failed(stop, log)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def judge(nb: Path, decl: Declaration, routine: Path, serial: Path, backend: str,
          timeout_s: float = 900.0, name: str | None = None) -> Verdict:
    """Judge a routine for a notebook on a backend against the problem's serial routine."""
    name = name or decl.name

    def work(folder: Path, deadline: float, log: list[str]) -> Verdict:
        _build(folder, name, backend, "driver")
        _build(folder, name, backend, "serial_driver")
        log.append(_against_python(nb, decl, folder, "driver", backend, _test_sizes(decl),
                                   deadline))  # fmt: skip
        return _benchmark(nb, folder, decl, backend, deadline, log)

    return _in_fresh_folder(decl, name, backend, routine, serial, timeout_s, work)


def correct_at(nb: Path, decl: Declaration, routine: Path, backend: str,
               sizes: list[int] | None = None, name: str | None = None,
               timeout_s: float = 900.0) -> Verdict:
    """Check a routine against the notebook's Python at the given sizes, or at the test sizes
    and two drawn at random when none are given.

    The backend "serial" checks a serial routine, as step 1 does before it accepts one.
    """
    name = name or decl.name
    serial = backend == "serial"
    program = "serial_driver" if serial else "driver"

    def work(folder: Path, deadline: float, log: list[str]) -> Verdict:
        _build(folder, name, backend, program)
        log.append(_against_python(nb, decl, folder, program, backend, sizes or _test_sizes(decl),
                                   deadline))  # fmt: skip
        return Verdict(True, "none")

    return _in_fresh_folder(decl, name, backend, None if serial else routine,
                            routine if serial else None, timeout_s, work)  # fmt: skip


def run_routine(decl: Declaration, routine: Path, backend: str, name: str,
                calls: list[tuple[Path, Path, int]], timeout_s: float = 600.0) -> Verdict:
    """Build a routine under the rule, and run it once on each (inputs, results, workers).

    Each call writes its results where it says. The verdict is correct when the routine was
    built and every call ran; it says nothing about the results themselves.
    """

    def work(folder: Path, deadline: float, log: list[str]) -> Verdict:
        _build(folder, name, backend, "driver")
        for inputs, results, workers in calls:
            limit = min(deadline, time.monotonic() + TEST_CALL_S)
            driver = Driver(folder, "driver", backend, workers, limit, stage="test")
            try:
                driver.call(inputs, results, timed=False)
            finally:
                driver.close()
        return Verdict(True, "none")

    return _in_fresh_folder(decl, name, backend, routine, None, timeout_s, work)


def _compared(slower: list[float], faster: list[float]) -> tuple[float, bool]:
    """Compare two programs' times on the same inputs, input by input.

    Returns the median of the ratios ``slower / faster`` and whether ``faster`` won on every
    input, which chance alone would give once in two to the power of the number of inputs.
    """
    pairs = list(zip(slower, faster, strict=True))
    return statistics.median(s / f for s, f in pairs), all(f < s for s, f in pairs)


def _check_against_serial(decl: Declaration, inputs: Path, result: Path,
                          sizes: dict[str, int]) -> None:
    """Compare a result with the serial routine's for the same input, then delete it."""
    wrong = compare(decl, read_results(result, decl, sizes),
                    read_results(inputs.with_suffix(".serial"), decl, sizes))  # fmt: skip
    result.unlink()
    if wrong:
        raise _Stop("benchmark", "wrong at the benchmark size: " + "; ".join(wrong))


def _benchmark(nb: Path, folder: Path, decl: Declaration, backend: str, deadline: float,
               log: list[str]) -> Verdict:
    """Time the routine against the serial routine, and at one worker, at the benchmark size.

    Each input is made just before its calls, and deleted with its results once they have been
    compared, so that the folder never holds more than one input and its results.
    """
    def fresh(i: int) -> tuple[Path, dict[str, int]]:
        path = folder / f"bench{i}.in"
        call = (decl.benchmark_size, secrets.randbits(32), path)
        return path, notebook.write_input_files(nb, [call], _left(deadline))[0]

    load_before = os.getloadavg()[0]
    inputs, sizes = fresh(0)
    most = max(_widths(backend, _bytes_of(decl, sizes)))
    runs = {"routine": Driver(folder, "driver", backend, most, deadline),
            "serial": Driver(folder, "serial_driver", "serial", 1, deadline)}  # fmt: skip
    if backend != "cuda" and most > 1:
        runs["one worker"] = Driver(folder, "driver", backend, 1, deadline)
    times: dict[str, list[float]] = {key: [] for key in runs}
    try:
        for i in range(TIMED_CALLS + 1):
            if i > 0:
                inputs, sizes = fresh(i)
            results = {key: inputs.with_suffix("." + key.replace(" ", "-")) for key in runs}
            for key, driver in runs.items():
                spent = driver.call(inputs, results[key], timed=i > 0)
                if spent is not None:
                    times[key].append(spent)
            for key in [k for k in runs if k != "serial"]:
                try:
                    _check_against_serial(decl, inputs, results[key], sizes)
                except _Stop as stop:
                    raise _Stop("benchmark", f"{stop} ({key})") from None
            inputs.unlink()
            results["serial"].unlink()
    finally:
        for driver in runs.values():
            driver.close()
    routine, serial = times["routine"], times["serial"]
    speedup, faster = _compared(serial, routine)
    scaling, parallel = None, None
    if "one worker" in times:
        scaling, parallel = _compared(times["one worker"], routine)
    elif backend == "cuda":
        called = subprocess.run(["nm", "-u", "routine.o"], cwd=folder, capture_output=True,
                                text=True).stdout  # fmt: skip
        parallel = "cudaLaunchKernel" in called or "__cudaPushCallConfiguration" in called
    workers = "on the graphics card" if backend == "cuda" else f"with {most} workers"
    log.append(
        f"correct at the benchmark size {decl.benchmark_size}; median times: routine "
        f"{statistics.median(routine):.4f} s {workers}, serial "
        f"{statistics.median(serial):.4f} s; speed-up {speedup:.2f}"
        + ("" if scaling is None else f"; scaling {scaling:.2f}")
    )
    for key, calls in times.items():
        log.append(f"timed calls, {key}: " + ", ".join(f"{s:.4f}" for s in calls) + " s")
    # Another program that runs meanwhile can slow any call, so the load is kept for checking.
    log.append(f"the machine's load average was {load_before:.1f} before the benchmark and "
               f"{os.getloadavg()[0]:.1f} after it")
    return Verdict(True, "none", speedup, faster, scaling, parallel)
