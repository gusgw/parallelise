"""Tests for harness_common.tools.run_tests."""
import sys

from harness_common.tools import run_tests


def test_run_tests_pass(tmp_path):
    r = run_tests.run_tests(str(tmp_path), cmd=[sys.executable, "-c", "print('x')"])
    assert r.ok
    assert "PASS" in run_tests.summarize(r)


def test_run_tests_fail_summary(tmp_path):
    r = run_tests.run_tests(str(tmp_path), cmd=[sys.executable, "-c", "import sys; sys.exit(1)"])
    assert not r.ok
    assert "FAIL" in run_tests.summarize(r)
