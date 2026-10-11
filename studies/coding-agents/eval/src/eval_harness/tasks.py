"""Eval tasks and their automatic checks.

The Task type and the check helpers live here; the concrete task instances are
loaded from eval_harness.tasks_local if that module is present. Without it TASKS
is empty, which is how the unit tests run.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from harness_common.targets import Target
from harness_common.tools import fs, proc


@dataclass
class CheckResult:
    passed: bool
    detail: str


@dataclass
class Task:
    key: str
    target: str
    kind: str          # debug | test | document | review | optimise
    prompt: str
    check: Callable[[Target], CheckResult] | None = None
    reset: list[list[str]] = field(default_factory=lambda: [["git", "checkout", "--", "."]])


def check_tests_pass() -> Callable[[Target], CheckResult]:
    """Check: the target's test suite passes."""
    def _check(target: Target) -> CheckResult:
        r = proc.run(target.test_cmd, target.path)
        return CheckResult(r.ok, "tests " + ("passed" if r.ok else f"failed (rc={r.returncode})"))
    return _check


def check_file_contains(path: str, needle: str) -> Callable[[Target], CheckResult]:
    """Check: a file under the target exists and contains needle (case-insensitive)."""
    def _check(target: Target) -> CheckResult:
        try:
            text = fs.read_file(target.path, path)
        except OSError:
            return CheckResult(False, f"missing {path}")
        ok = needle.lower() in text.lower()
        return CheckResult(ok, f"{'found' if ok else 'absent'}: '{needle}' in {path}")
    return _check


def check_frozen_test(restore=("tests",)) -> Callable[[Target], CheckResult]:
    """Run the suite, but first restore the given (test) paths from the worktree's
    HEAD — so an agent cannot 'pass' by deleting or weakening the frozen spec. The
    agent's source edits are left intact; only the named test paths are reset."""
    def _check(target: Target) -> CheckResult:
        for path in restore:
            proc.run(["git", "checkout", "--", path], target.path)
        r = proc.run(target.test_cmd, target.path)
        return CheckResult(r.ok, "tests " + ("passed" if r.ok else f"failed (rc={r.returncode})"))
    return _check


def _load_tasks() -> list["Task"]:
    """Load the study's tasks if the module that lists them is present, else none."""
    try:
        from eval_harness import tasks_local
    except ImportError:
        return []
    return list(tasks_local.TASKS)


TASKS = _load_tasks()
