"""Tests for eval_harness.tasks (Task type + check helpers; no task data)."""
import sys

from harness_common.targets import Target

from eval_harness import tasks


def test_task_construction():
    t = tasks.Task("k", "demo", "debug", "do the thing")
    assert t.key == "k" and t.target == "demo" and t.prompt
    assert t.reset  # default reset present


def test_load_tasks_tolerates_absent_local():
    # Without the tasks module the list is empty, never an error.
    assert isinstance(tasks._load_tasks(), list)


def test_check_file_contains(tmp_path):
    (tmp_path / "REVIEW.md").write_text("Notes about the storage layer.")
    target = Target(name="t", path=str(tmp_path))
    assert tasks.check_file_contains("REVIEW.md", "storage")(target).passed is True
    assert tasks.check_file_contains("REVIEW.md", "absent-word")(target).passed is False
    assert tasks.check_file_contains("MISSING.md", "x")(target).passed is False


def test_check_tests_pass(tmp_path):
    ok_target = Target(name="ok", path=str(tmp_path),
                       test_cmd=[sys.executable, "-c", "import sys; sys.exit(0)"])
    bad_target = Target(name="bad", path=str(tmp_path),
                        test_cmd=[sys.executable, "-c", "import sys; sys.exit(1)"])
    assert tasks.check_tests_pass()(ok_target).passed is True
    assert tasks.check_tests_pass()(bad_target).passed is False
