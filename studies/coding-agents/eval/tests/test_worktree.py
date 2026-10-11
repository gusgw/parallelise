"""Tests for eval_harness.worktree (uses a real temp git repo)."""
import os
import subprocess
from pathlib import Path

from eval_harness import worktree


def _make_repo(tmp_path):
    d = str(tmp_path / "repo")
    os.makedirs(d)
    subprocess.run(["git", "init", "-q"], cwd=d, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=d, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=d, check=True)
    (Path(d) / "f.txt").write_text("hello\n")
    subprocess.run(["git", "add", "."], cwd=d, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=d, check=True)
    return d


def test_worktree_isolates_edits(tmp_path):
    repo = _make_repo(tmp_path)
    wt = worktree.add_worktree(repo)
    try:
        assert os.path.isdir(wt)
        assert (Path(wt) / "f.txt").read_text() == "hello\n"     # fresh copy at HEAD
        (Path(wt) / "f.txt").write_text("changed\n")             # edit in the worktree
        assert "changed" in worktree.diff(wt)                    # diff sees it
        assert (Path(repo) / "f.txt").read_text() == "hello\n"   # main repo untouched
    finally:
        worktree.remove_worktree(repo, wt)
    assert not os.path.isdir(wt)
