"""Per-run git worktree isolation for best-of-N eval runs.

Each attempt gets a fresh detached worktree of the target at HEAD, so concurrent
attempts don't clobber each other and the user's real working tree is never
touched. The target's .venv is symlinked in so its test command still works.
"""
from __future__ import annotations

import os
import tempfile
import threading

from harness_common.tools import proc

# git serialises its own worktree metadata, but concurrent add/remove can still
# collide on the lock; guard those (cheap) mutations so attempts can otherwise run
# in parallel.
_GIT_LOCK = threading.Lock()


def add_worktree(target_path: str, timeout: float = 60.0) -> str:
    """Create a detached worktree of target_path at HEAD; symlink .venv. Return its path."""
    wt = tempfile.mkdtemp(prefix="eval-wt-")
    os.rmdir(wt)  # git worktree add wants a non-existent path
    with _GIT_LOCK:
        proc.run(["git", "-C", target_path, "worktree", "add", "--detach", wt, "HEAD"],
                 target_path, timeout)
    venv = os.path.join(target_path, ".venv")
    if os.path.isdir(venv):
        os.symlink(venv, os.path.join(wt, ".venv"))
    return wt


def remove_worktree(target_path: str, wt: str, timeout: float = 60.0) -> None:
    """Remove a worktree created by add_worktree (best-effort)."""
    with _GIT_LOCK:
        proc.run(["git", "-C", target_path, "worktree", "remove", "--force", wt], target_path, timeout)


def diff(wt: str, timeout: float = 60.0) -> str:
    """Return the git diff of the changes an agent made inside the worktree."""
    return proc.run(["git", "-C", wt, "diff"], wt, timeout).output
