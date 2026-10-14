"""The check: an independent judgement of a finished work folder.

A work folder passes only if its trusted files are unchanged since assembly, it compiles
with warnings treated as errors, its unit test passes at one worker and at several, its
benchmark runs, and the benchmark's speed-up figures survive the checks in ``bench_check``.
The check never trusts what the agent says about its own work; it rebuilds from
scratch and reads only what the trusted benchmark wrote.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from parallelise.assemble import TRUSTED_MANIFEST, file_digest, kernel_filename
from parallelise.bench_check import (
    DEFAULT_CUDA_FLOOR,
    DEFAULT_FLOOR,
    BenchVerdict,
    check_bench_csv,
    check_bench_cuda,
)

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


def _find_kernel_name(work_dir: Path) -> str | None:
    """Work out the kernel name from the Makefile, the notebook or the test target."""
    makefile = work_dir / "Makefile"
    text = makefile.read_text(encoding="utf-8", errors="replace") if makefile.is_file() else ""
    match = re.search(r"^KERNEL\s*[:?]?=\s*(\w+)", text, re.MULTILINE)
    if match:
        return match.group(1)
    notebooks = sorted(work_dir.glob("*.ipynb"))
    if len(notebooks) == 1:
        return notebooks[0].stem
    match = re.search(r"^test_(\w+)\s*:", text, re.MULTILINE)
    return match.group(1) if match else None


def _makefile_floor(work_dir: Path, backend: str) -> float:
    """Return the speed-up floor the Makefile asks for, or the default for the backend."""
    text = (work_dir / "Makefile").read_text(encoding="utf-8", errors="replace")
    match = re.search(r"SPEEDUP_FLOOR=([0-9.]+)", text)
    if match:
        return float(match.group(1))
    return DEFAULT_CUDA_FLOOR if backend == "cuda" else DEFAULT_FLOOR


def _check_recipe(work_dir: Path) -> dict[int, str]:
    """Return the recipe lines of the ``check`` target, keyed by line number."""
    lines = (work_dir / "Makefile").read_text(encoding="utf-8", errors="replace").splitlines()
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


def _classify_check_failure(work_dir: Path, output: str) -> str:
    """Decide which stage of ``make check`` failed from the line that make reports."""
    match = re.search(r"\[[^\]:]*:(\d+): check\]", output)
    recipe = _check_recipe(work_dir)
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


def _tampered_files(work_dir: Path) -> list[str]:
    """Return a description of each trusted file that changed since the folder was built.

    The list is empty when every file matches the digests in ``TRUSTED_MANIFEST``. A missing
    or unreadable manifest is reported as a problem, because without it nothing can be
    checked.
    """
    manifest = work_dir / TRUSTED_MANIFEST
    try:
        expected = json.loads(manifest.read_text(encoding="utf-8"))["sha256"]
        if not isinstance(expected, dict) or not expected:
            raise ValueError("no digests recorded")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [f"{TRUSTED_MANIFEST} is missing or unreadable ({exc})"]
    problems = []
    for filename, digest in sorted(expected.items()):
        path = work_dir / filename
        if not path.is_file():
            problems.append(f"{filename} has been removed")
        elif file_digest(path) != digest:
            problems.append(f"{filename} has been changed")
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


def run_check(
    work_dir: Path,
    backend: str,
    timeout_s: int = 300,
    speedup_floor: float | None = None,
) -> CheckResult:
    """Judge a work folder; the whole judgement shares one time limit of ``timeout_s``.

    The speed-up floor normally comes from the problem's Makefile. When ``speedup_floor``
    is given, the benchmark table is checked against it as well, so a stricter floor than
    the Makefile's takes effect (a looser one cannot, because the Makefile's own check
    runs first).
    """
    work_dir = Path(work_dir).resolve()
    record: list[str] = []

    def result(stage: str, best: float | None = None) -> CheckResult:
        passed = stage == "none"
        record.append(f"CHECK {'PASS' if passed else 'FAIL'} ({stage}) {work_dir}")
        return CheckResult(passed, stage, best, "\n".join(record))

    changed = _tampered_files(work_dir)
    if changed:
        record.append("[check] trusted files do not match the digests written at assembly:")
        record.extend(f"[check]   {line}" for line in changed)
        return result("tampered")

    name = _find_kernel_name(work_dir)
    if name is None or not (work_dir / "Makefile").is_file():
        record.append("[check] no Makefile or kernel name found in the work folder")
        return result("no-files")
    needed = [kernel_filename(name, backend), f"test_{name}.c"]
    missing = [f for f in needed if not (work_dir / f).is_file()]
    if missing:
        record.append("[check] missing files: " + ", ".join(missing))
        return result("no-files")

    env = dict(os.environ)
    if backend in STRICT_CFLAGS:
        env["CFLAGS"] = STRICT_CFLAGS[backend]
    # The Makefiles run "python3 check_bench.py"; make sure that finds this interpreter
    # and the package, wherever the check was started from.
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    package_parent = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = package_parent + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("SPEEDUP_FLOOR", None)

    deadline = time.monotonic() + timeout_s
    best: float | None = None
    try:
        record.append("[check] build with warnings as errors")
        _run(["make", "clean"], work_dir, env, deadline)
        code, out = _run(["make", f"test_{name}", "bench"], work_dir, env, deadline)
        record.append(out)
        if code != 0:
            return result("build")

        record.append("[check] unit test at one worker and at several, then benchmark")
        code, out = _run(["make", "check"], work_dir, env, deadline)
        record.append(out)
        csv_path = work_dir / "bench.csv"
        floor = speedup_floor if speedup_floor is not None else _makefile_floor(work_dir, backend)
        verdict: BenchVerdict | None = None
        if csv_path.is_file():
            checker = check_bench_cuda if backend == "cuda" else check_bench_csv
            verdict = checker(csv_path, floor)
            best = verdict.speedup_best
        if code != 0:
            return result(_classify_check_failure(work_dir, out), best)
        if verdict is None:
            record.append("[check] the benchmark wrote no bench.csv")
            return result("benchmark")
        if not verdict.ok:
            record.append("[check] " + verdict.message)
            return result("benchmark", best)
    except _Timeout as exc:
        if exc.args and exc.args[0]:
            record.append(str(exc.args[0]))
        record.append(f"[check] time limit of {timeout_s} s reached")
        return result("timeout")
    except OSError as exc:
        record.append(f"[check] could not run the build tools: {exc}")
        return result("build")
    return result("none", best)
