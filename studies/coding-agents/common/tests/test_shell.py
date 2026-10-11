"""Tests for harness_common.tools.shell."""
import pytest

from harness_common.tools import shell


def test_empty_command_raises(tmp_path):
    with pytest.raises(ValueError):
        shell.run_command(str(tmp_path), [])


def test_disallowed_program_raises(tmp_path):
    with pytest.raises(ValueError):
        shell.run_command(str(tmp_path), ["rm", "-rf", "x"])


def test_allowed_program_runs(tmp_path):
    r = shell.run_command(str(tmp_path), ["ls"])
    assert r.ok
