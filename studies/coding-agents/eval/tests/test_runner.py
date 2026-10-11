"""Tests for eval_harness.runner (with a fake command runner — no agents/targets)."""
from harness_common.targets import Target
from harness_common.tools.proc import ProcResult

from eval_harness import runner
from eval_harness.harnesses import Harness
from eval_harness.tasks import CheckResult, Task


def _fake_run(record):
    def fake(cmd, cwd, timeout=900.0):
        record.append((cmd, cwd))
        return ProcResult(True, 0, "some output")
    return fake


def test_run_one_with_check_times_and_runs(tmp_path):
    target = Target(name="t", path=str(tmp_path))
    task = Task("k", "t", "debug", "do it",
                check=lambda tg: CheckResult(True, "ok"), reset=[["reset-cmd"]])
    record = []
    clock = iter([1.0, 4.0])
    res = runner.run_one(Harness("h", "proj", "mod"), task, "/root", target,
                         run=_fake_run(record), now=lambda: next(clock))
    assert res.passed is True and res.detail == "ok"
    assert res.seconds == 3.0
    assert res.harness == "h" and res.task == "k"
    assert len(record) == 2          # reset + harness command both ran


def test_run_one_without_check(tmp_path):
    target = Target(name="t", path=str(tmp_path))
    task = Task("k", "t", "document", "doc", check=None, reset=[])
    res = runner.run_one(Harness("h", "proj", "mod"), task, "/root", target,
                         run=lambda *a, **k: ProcResult(True, 0, "x"), now=lambda: 0.0)
    assert res.passed is None and "manual" in res.detail


def test_run_one_records_errors_instead_of_raising(tmp_path):
    target = Target(name="t", path=str(tmp_path))
    task = Task("k", "t", "debug", "p", check=None, reset=[])

    def boom(*a, **k):
        raise FileNotFoundError("no target")

    res = runner.run_one(Harness("h", "proj", "mod"), task, "/root", target,
                         run=boom, now=lambda: 0.0)
    assert res.passed is False and "run error" in res.detail


def test_run_eval_loops_over_pairs(monkeypatch):
    from harness_common import targets as targets_mod
    monkeypatch.setattr(targets_mod, "get_target", lambda name: Target(name=name, path="/real"))
    task = Task("k", "demo", "debug", "p",
                check=lambda tg: CheckResult(True, "ok"), reset=[])
    res = runner.run_eval([Harness("h", "proj", "mod")], [task], "/root",
                          run=lambda *a, **k: ProcResult(True, 0, "o"), now=lambda: 0.0)
    assert len(res) == 1 and res[0].passed is True


def test_render_markdown_shows_all_statuses():
    rows = [
        runner.RunResult("h", "k1", True, "ok", 3.0, "x"),
        runner.RunResult("h", "k2", False, "bad", 1.0, "y"),
        runner.RunResult("h", "k3", None, "manual", 2.0, "z"),
    ]
    md = runner.render_markdown(rows)
    assert "PASS" in md and "FAIL" in md and "skip" in md


# --- best-of-N (Phase 8.2): worktree-isolated attempts ---


def _attempt_fakes():
    rec = {"removed": None, "env_target": None, "env_variant": None, "added_for": None}

    def add(path):
        rec["added_for"] = path
        return "/wt"

    def remove(path, wt):
        rec["removed"] = wt

    def get_diff(wt):
        return "DIFF@" + wt

    def run(cmd, cwd, timeout=900.0, env=None):
        rec["env_target"] = (env or {}).get("HARNESS_TARGET_PATH")
        rec["env_variant"] = (env or {}).get("VARIANT")
        return ProcResult(True, 0, 'out\nTELEMETRY {"turns": 5, "tokens": 4200}')

    return add, remove, get_diff, run, rec


def test_run_attempt_isolates_in_worktree():
    target = Target(name="t", path="/real")
    # the check passes only if run against the worktree path -> proves isolation
    task = Task("k", "t", "debug", "p",
                check=lambda tg: CheckResult(tg.path == "/wt", "checked " + tg.path), reset=[])
    add, remove, get_diff, run, rec = _attempt_fakes()
    res = runner.run_attempt(Harness("h", "proj", "mod", env={"VARIANT": "x"}),
                             task, "/root", target,
                             run=run, now=lambda: 0.0, add=add, remove=remove, get_diff=get_diff)
    assert res.passed is True
    assert res.diff == "DIFF@/wt"
    assert res.turns == 5 and res.tokens == 4200   # telemetry parsed from stdout
    assert rec["env_target"] == "/wt"      # the agent was pointed at the worktree
    assert rec["env_variant"] == "x"       # harness extra env merged in
    assert rec["added_for"] == "/real"     # worktree made from the real target
    assert rec["removed"] == "/wt"         # and cleaned up


def test_run_attempt_cleans_up_on_error():
    target = Target(name="t", path="/real")
    task = Task("k", "t", "debug", "p", check=None, reset=[])
    rec = {"removed": None}

    def run(cmd, cwd, timeout=900.0, env=None):
        raise FileNotFoundError("boom")

    res = runner.run_attempt(Harness("h", "proj", "mod"), task, "/root", target,
                             run=run, now=lambda: 0.0, add=lambda p: "/wt",
                             remove=lambda p, w: rec.__setitem__("removed", w),
                             get_diff=lambda w: "")
    assert res.passed is False and "run error" in res.detail
    assert rec["removed"] == "/wt"          # cleaned up even when the agent errors


def test_best_of_n_aggregates(monkeypatch):
    from harness_common import targets as targets_mod
    monkeypatch.setattr(targets_mod, "get_target", lambda name: Target(name=name, path="/real"))
    outcomes = iter([True, False, False])   # 1 of 3 attempts passes
    task = Task("k", "demo", "debug", "p",
                check=lambda tg: CheckResult(next(outcomes), "c"), reset=[])
    r = runner.best_of_n(Harness("h", "proj", "mod"), task, "/root", n=3, concurrency=1,
                         run=lambda *a, **k: ProcResult(True, 0, "o"), now=lambda: 0.0,
                         add=lambda p: "/wt", remove=lambda p, w: None, get_diff=lambda w: "d")
    assert r.n == 3 and r.successes == 1 and r.rate() == "1/3"
    assert r.best().passed is True


def test_render_bestofn_shows_rate_and_breakdown():
    a = runner.AttemptResult(True, "fixed it", 5.0, "d1", "passed")
    b = runner.AttemptResult(False, "nope", 3.0, "", "tests-fail")
    md = runner.render_bestofn([runner.BestOfN("h", "k", 2, [a, b])])
    assert "1/2" in md and "1x passed" in md and "1x tests-fail" in md and "| h | k |" in md


def test_classify_reason_taxonomy():
    assert runner.classify_reason(True, "", "", "") == "passed"
    assert runner.classify_reason(False, "GraphRecursionError: ...", "", "d") == "recursion-limit"
    assert runner.classify_reason(False, "pydantic_ai...UsageLimitExceeded", "", "d") == "usage-limit"
    assert runner.classify_reason(False, "(stopped: reached max_steps=20)", "", "d") == "step-limit"
    assert runner.classify_reason(False, "done", "", "") == "no-edit"
    assert runner.classify_reason(False, "done", "tests failed (rc=2)", "d") == "still-broken"
    assert runner.classify_reason(False, "done", "tests failed (rc=1)", "d") == "tests-fail"


def test_write_artifacts_one_line_per_attempt(tmp_path):
    import json
    res = [runner.BestOfN("h", "k", 2, [
        runner.AttemptResult(True, "ok", 5.0, "diff", "passed", "tail", 8, 9000),
        runner.AttemptResult(False, "no", 3.0, "", "no-edit", "tail2", 2, 500)])]
    path = str(tmp_path / "art.jsonl")
    runner.write_artifacts(res, path)
    rows = [json.loads(line) for line in open(path)]
    assert len(rows) == 2
    assert rows[0]["passed"] is True and rows[0]["reason"] == "passed"
    assert rows[0]["turns"] == 8 and rows[0]["tokens"] == 9000
    assert rows[1]["reason"] == "no-edit"
