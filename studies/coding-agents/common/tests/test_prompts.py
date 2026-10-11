"""Tests for harness_common.prompts."""
from harness_common import prompts
from harness_common.tools import fs


def test_strategy_mentions_key_actions():
    s = prompts.STRATEGY
    assert "relative" in s and "edit_file" in s and "grep" in s


def test_with_repo_listing_includes_files(tmp_path):
    fs.write_file(str(tmp_path), "a.txt", "x")
    (tmp_path / "sub").mkdir()
    out = prompts.with_repo_listing("do it", str(tmp_path))
    assert out.startswith("do it")
    assert "a.txt" in out and "sub/" in out
    assert "relative to here" in out
