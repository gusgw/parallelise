"""Run a subprocess and capture its combined output as a ProcResult."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class ProcResult:
    ok: bool
    returncode: int
    output: str


def run(cmd: list[str], cwd: str, timeout: float = 600.0, env: dict | None = None) -> ProcResult:
    """Run cmd (list form, no shell) in cwd; ok is True when returncode is 0.

    env=None inherits the parent environment; pass a dict to override (per-process,
    so concurrent runs don't race on os.environ).
    """
    proc = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env
    )
    return ProcResult(proc.returncode == 0, proc.returncode, proc.stdout + proc.stderr)
