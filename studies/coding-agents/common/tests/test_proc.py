"""Tests for harness_common.tools.proc."""
import sys

from harness_common.tools import proc


def test_run_success(tmp_path):
    r = proc.run([sys.executable, "-c", "print('ok')"], str(tmp_path))
    assert r.ok and r.returncode == 0
    assert "ok" in r.output


def test_run_failure(tmp_path):
    r = proc.run([sys.executable, "-c", "import sys; sys.exit(3)"], str(tmp_path))
    assert not r.ok and r.returncode == 3


def test_run_passes_env(tmp_path):
    import os
    env = dict(os.environ)
    env["HARNESS_TEST_X"] = "zzz9"
    r = proc.run([sys.executable, "-c", "import os; print(os.environ.get('HARNESS_TEST_X', ''))"],
                 str(tmp_path), env=env)
    assert "zzz9" in r.output
