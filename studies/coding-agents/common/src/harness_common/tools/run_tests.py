"""Run a target's test command and summarise pass/fail for the model."""
from __future__ import annotations

from harness_common.tools import proc


def run_tests(cwd: str, cmd: list[str] | None = None, timeout: float = 600.0) -> proc.ProcResult:
    """Run a test command (default: pytest -q) in cwd."""
    return proc.run(cmd or ["python", "-m", "pytest", "-q"], cwd, timeout)


def summarize(result: proc.ProcResult, max_lines: int = 25, max_chars: int = 3000) -> str:
    """Compact summary: a status line plus the (bounded) tail of the output."""
    status = "PASS" if result.ok else "FAIL (rc=" + str(result.returncode) + ")"
    tail = "\n".join(result.output.splitlines()[-max_lines:])
    if len(tail) > max_chars:
        tail = tail[-max_chars:]
    return status + "\n" + tail
