"""Run a harness over an eval task and score it."""
from __future__ import annotations

import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from harness_common import targets, telemetry
from harness_common.targets import Target
from harness_common.tools import proc

from eval_harness import worktree
from eval_harness.harnesses import Harness, harness_command
from eval_harness.tasks import Task


@dataclass
class RunResult:
    harness: str
    task: str
    passed: bool | None      # None = no auto-check (manual rubric)
    detail: str
    seconds: float
    output_tail: str


def _tail(text: str, n: int = 20) -> str:
    return "\n".join(text.splitlines()[-n:])


def run_one(harness: Harness, task: Task, harness_root: str, target, *,
            run=proc.run, now=time.monotonic, timeout: float = 900.0) -> RunResult:
    """Reset the target, run the harness on the task, then apply the task's check.

    Errors (missing target, agent crash, check failure) are recorded as a failed
    result rather than raised, so one bad pair doesn't abort the whole eval.
    """
    try:
        for cmd in task.reset:
            run(cmd, target.path)
    except Exception as exc:  # record, don't abort the eval
        return RunResult(harness.name, task.key, False, f"reset error: {exc}", 0.0, "")

    start = now()
    try:
        result = run(harness_command(harness, task),
                     os.path.join(harness_root, harness.project_dir), timeout)
    except Exception as exc:
        return RunResult(harness.name, task.key, False, f"run error: {exc}", now() - start, "")
    seconds = now() - start

    if task.check is None:
        return RunResult(harness.name, task.key, None, "no auto-check (manual rubric)",
                         seconds, _tail(result.output))
    try:
        chk = task.check(target)
    except Exception as exc:
        return RunResult(harness.name, task.key, False, f"check error: {exc}", seconds, _tail(result.output))
    return RunResult(harness.name, task.key, chk.passed, chk.detail, seconds, _tail(result.output))


def run_eval(harness_list, task_list, harness_root, *, run=proc.run, now=time.monotonic):
    """Run every (harness, task) pair; return a flat list of RunResult."""
    results = []
    for task in task_list:
        target = targets.get_target(task.target)
        for harness in harness_list:
            results.append(run_one(harness, task, harness_root, target, run=run, now=now))
    return results


def render_markdown(results) -> str:
    """Render results as a markdown table."""
    lines = ["| harness | task | result | secs |", "|---|---|---|---|"]
    for r in results:
        status = "skip" if r.passed is None else ("PASS" if r.passed else "FAIL")
        lines.append(f"| {r.harness} | {r.task} | {status} | {r.seconds:.0f} |")
    return "\n".join(lines)


# --- best-of-N (Phase 8.2): N isolated worktree attempts per (harness, task) ---


@dataclass
class AttemptResult:
    passed: bool | None      # None = no auto-check
    detail: str
    seconds: float
    diff: str                # the change the agent made, from the worktree
    reason: str = ""         # classified terminal reason (see classify_reason)
    output_tail: str = ""    # tail of the agent's stdout
    turns: int = -1          # model round-trips the agent made (-1 = unknown)
    tokens: int = -1         # total tokens the agent used (-1 = unknown)


def classify_reason(passed, agent_output: str, check_detail: str, diff: str) -> str:
    """A short tag for *why* an attempt ended as it did — the failure taxonomy.

    Distinguishes ran-out-of-budget reasons (timeout / recursion / usage / step
    limits) from finished-but-wrong states (no edit / still uncollectable /
    tests fail), so a censoring timeout is never confused with incapability.
    """
    if passed:
        return "passed"
    low = (agent_output or "").lower()
    if "timeout" in low or "timed out" in low:
        return "timeout"
    if "graphrecursionerror" in low or "recursion limit" in low:
        return "recursion-limit"
    if "usagelimitexceeded" in low or "request_limit" in low:
        return "usage-limit"
    if "reached max_steps" in low:
        return "step-limit"
    if "repeated the same tool call" in low:
        return "repeat-abort"
    if not (diff or "").strip():
        return "no-edit"                 # finished without changing anything
    d = (check_detail or "").lower()
    if "rc=2" in d:
        return "still-broken"            # edited, but the suite still won't collect
    if "rc=1" in d:
        return "tests-fail"              # collects now, but a test fails
    return "other"


@dataclass
class BestOfN:
    harness: str
    task: str
    n: int
    attempts: list

    @property
    def successes(self) -> int:
        return sum(1 for a in self.attempts if a.passed is True)

    def rate(self) -> str:
        return str(self.successes) + "/" + str(self.n)

    def reason_counts(self) -> dict:
        """Map terminal reason -> count, most common first."""
        from collections import Counter
        return dict(Counter(a.reason for a in self.attempts).most_common())

    def best(self):
        """The first passing attempt, else the first attempt (for the diff/detail)."""
        for a in self.attempts:
            if a.passed is True:
                return a
        return self.attempts[0] if self.attempts else None


def run_attempt(harness: Harness, task: Task, harness_root: str, base_target, *,
                run=proc.run, now=time.monotonic, add=worktree.add_worktree,
                remove=worktree.remove_worktree, get_diff=worktree.diff,
                timeout: float = 900.0) -> AttemptResult:
    """Run one attempt in a fresh worktree at HEAD; check there; capture the diff."""
    wt = add(base_target.path)
    try:
        env = dict(os.environ)
        env["HARNESS_TARGET_PATH"] = wt          # the agent resolves the target to here
        env.update(harness.env)                  # e.g. AGENT_LANGGRAPH_VARIANT
        start = now()
        try:
            res = run(harness_command(harness, task),
                      os.path.join(harness_root, harness.project_dir), timeout, env=env)
            agent_out = _tail(res.output)
            tel = telemetry.parse(res.output)
        except Exception as exc:
            msg = "run error: " + str(exc)
            reason = "timeout" if "timeout" in str(exc).lower() or "timed out" in str(exc).lower() else "run-error"
            return AttemptResult(False, msg, now() - start, "", reason, msg)
        seconds = now() - start
        turns, tokens = tel.get("turns", -1), tel.get("tokens", -1)
        wt_target = Target(name=base_target.name, path=wt,
                           apps=list(base_target.apps), test_cmd=list(base_target.test_cmd))
        diff = get_diff(wt)
        if task.check is None:
            return AttemptResult(None, "no auto-check (manual rubric)", seconds, diff,
                                 "manual", agent_out, turns, tokens)
        try:
            chk = task.check(wt_target)
        except Exception as exc:
            return AttemptResult(False, "check error: " + str(exc), seconds, diff,
                                 "check-error", agent_out, turns, tokens)
        reason = classify_reason(chk.passed, agent_out, chk.detail, diff)
        return AttemptResult(chk.passed, chk.detail, seconds, diff, reason, agent_out, turns, tokens)
    finally:
        remove(base_target.path, wt)


def best_of_n(harness: Harness, task: Task, harness_root: str, n: int = 5,
              concurrency: int = 2, *, run=proc.run, now=time.monotonic,
              add=worktree.add_worktree, remove=worktree.remove_worktree,
              get_diff=worktree.diff) -> BestOfN:
    """Run a (harness, task) pair N times in isolated worktrees; aggregate."""
    base_target = targets.get_target(task.target)
    attempts = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(run_attempt, harness, task, harness_root, base_target,
                               run=run, now=now, add=add, remove=remove, get_diff=get_diff)
                   for _ in range(n)]
        for fut in futures:
            attempts.append(fut.result())
    return BestOfN(harness.name, task.key, n, attempts)


def render_bestofn(results) -> str:
    """Render best-of-N: success rate, median wall-clock/turns/tokens, reason breakdown."""
    lines = ["| harness | task | success | median s | med turns | med ktok | breakdown |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        secs = [a.seconds for a in r.attempts] or [0.0]
        turns = [a.turns for a in r.attempts if a.turns >= 0] or [0]
        ktok = [a.tokens / 1000 for a in r.attempts if a.tokens >= 0] or [0]
        breakdown = ", ".join(str(n) + "x " + reason for reason, n in r.reason_counts().items())
        lines.append("| " + r.harness + " | " + r.task + " | " + r.rate() +
                     " | " + format(statistics.median(secs), ".0f") +
                     " | " + format(statistics.median(turns), ".0f") +
                     " | " + format(statistics.median(ktok), ".1f") +
                     " | " + breakdown + " |")
    return "\n".join(lines)


def write_artifacts(results, path: str) -> None:
    """Write one JSON line per attempt (outcome, reason, timing, diff, output)."""
    import json
    with open(path, "w") as fh:
        for r in results:
            for i, a in enumerate(r.attempts):
                fh.write(json.dumps({
                    "harness": r.harness, "task": r.task, "attempt": i,
                    "passed": a.passed, "seconds": round(a.seconds, 1),
                    "turns": a.turns, "tokens": a.tokens,
                    "reason": a.reason, "detail": a.detail[:200],
                    "diff": a.diff[:4000], "output_tail": a.output_tail[:1200],
                }) + "\n")
