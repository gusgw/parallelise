"""Guarded shell runner: only allow-listed programs, fixed cwd, no shell=True."""
from __future__ import annotations

from harness_common.tools import proc

ALLOWED = {
    "git", "uv", "python", "python3", "pytest",
    "ls", "cat", "head", "tail", "grep", "find",
}


def run_command(cwd: str, cmd: list[str], timeout: float = 120.0) -> proc.ProcResult:
    """Run cmd (list form) in cwd if its program is allow-listed; else raise."""
    if not cmd:
        raise ValueError("empty command")
    if cmd[0] not in ALLOWED:
        raise ValueError("program not allowed: " + cmd[0])
    return proc.run(cmd, cwd, timeout)
