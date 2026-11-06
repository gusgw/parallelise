"""The tool surface shared by all three agent designs.

Every function returns text and never raises: a failure comes back as a string beginning
"error:", so that the model can read it and carry on. A missing optional file, for example,
must not end the run. Paths are confined to the work folder.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import signal
import subprocess
from pathlib import Path

from parallelise.bench_check import FLOOR_ROLE_VARIABLE, JUDGE_SPEED_VARIABLE
from parallelise.config import without_secrets

log = logging.getLogger("parallelise.agents.tools")

# Programs the model may start, besides the programs it has built itself (./name). The command
# is run as an argument list with no shell, so ';', '|' and '&&' are passed as plain arguments
# and cannot chain further commands. No interpreter is listed: python3, bash and sh would let
# the model run any code it likes. Note that make and the compilers can themselves start other
# programs, so this list narrows what the model can do but is not a sandbox.
ALLOWED_COMMANDS = frozenset(
    {"make", "gcc", "cc", "clang", "mpicc", "mpirun", "nvcc", "ls", "cat"}
)
COMMAND_TIMEOUT_S = 600
KILL_WAIT_S = 5
OUTPUT_TAIL_CHARS = 4000
DEFAULT_CHECK_COMMAND = ("make", "check")


def _resolve(work_dir: Path, relative: str) -> Path:
    """Return the absolute path of ``relative`` inside ``work_dir``, or raise ValueError."""
    root = work_dir.resolve()
    full = (root / relative).resolve()
    if full != root and root not in full.parents:
        raise ValueError(f"path {relative!r} is outside the work folder")
    return full


def _error(exc: BaseException) -> str:
    return f"error: {type(exc).__name__}: {exc}"


def notebook_text(notebook: dict) -> str:
    """Render the cells of a parsed notebook as readable text, in order."""
    parts = []
    for index, cell in enumerate(notebook.get("cells", [])):
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        parts.append(f"# --- cell {index} ({cell.get('cell_type', '?')}) ---\n{source}")
    return "\n\n".join(parts)


def read_notebook(work_dir: Path, path: str) -> str:
    """Read a Jupyter notebook in the work folder as plain text, markdown and code in order."""
    try:
        with open(_resolve(work_dir, path), encoding="utf-8") as handle:
            return notebook_text(json.load(handle))
    except (OSError, ValueError) as exc:
        return _error(exc)


def read_file(work_dir: Path, path: str) -> str:
    """Read a text file in the work folder."""
    try:
        return _resolve(work_dir, path).read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        return _error(exc)


def write_file(work_dir: Path, path: str, content: str) -> str:
    """Create or overwrite a file in the work folder, making folders as needed."""
    try:
        full = _resolve(work_dir, path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"
    except (OSError, ValueError) as exc:
        return _error(exc)


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill every process in the group that ``proc`` leads, and collect what it left."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass  # the whole group has already gone
    try:
        proc.communicate(timeout=KILL_WAIT_S)
    except subprocess.TimeoutExpired:
        # A descendant that left the group still holds the output pipes. Stop waiting for it.
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None:
                pipe.close()
        proc.wait()


def _agent_environment() -> dict[str, str]:
    """Return the environment for a command the agent runs: no secrets, and no setting that
    would change how its own ``make check`` judges the benchmark table."""
    env = without_secrets()
    for name in (FLOOR_ROLE_VARIABLE, JUDGE_SPEED_VARIABLE, "SPEEDUP_FLOOR"):
        env.pop(name, None)
    return env


def run_process(
    work_dir: Path, parts: list[str], timeout_s: int = COMMAND_TIMEOUT_S
) -> tuple[int, str]:
    """Run an argument list in the work folder; return (exit code, tail of the output).

    The program runs in a new session, so it and every child it starts form one process
    group, and when ``timeout_s`` seconds pass the whole group is killed: no child survives
    the command. The environment passed on has every variable that may hold a secret removed
    (see ``without_secrets``), so that a program the model compiled cannot read the gateway
    key from it. The settings that change how the benchmark table is judged, including the
    speed-up floor, are removed too, so that the agent's own ``make check`` always fails a
    missed floor and always judges speed, whatever the caller's shell says; only the check
    sets them, in its own judging folder. An exit code of -1 means the program could not be
    started or ran out of time.
    """
    try:
        proc = subprocess.Popen(
            parts,
            cwd=work_dir,
            env=_agent_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except OSError as exc:
        return -1, _error(exc)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        return -1, f"error: the command ran for more than {timeout_s} seconds and was stopped"
    except BaseException:
        # Ctrl-C and similar: the command is in a session of its own and would not see it.
        _kill_process_group(proc)
        raise
    return proc.returncode, (stdout + stderr)[-OUTPUT_TAIL_CHARS:]


def run_shell(work_dir: Path, command: str) -> str:
    """Run an allow-listed build or test command in the work folder and return its output.

    The command may name an allow-listed program, or a program inside the work folder
    written as ``./name``. A path that begins with ``./`` but leaves the work folder, such
    as ``./../../usr/bin/python3``, is refused.
    """
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return _error(exc)
    if not parts:
        return "error: empty command"
    if parts[0].startswith("./"):
        try:
            _resolve(work_dir, parts[0])
        except ValueError:
            return f"error: command not allowed: {parts[0]} is outside the work folder"
    elif Path(parts[0]).name not in ALLOWED_COMMANDS:
        return f"error: command not allowed: {parts[0]}"
    code, output = run_process(work_dir, parts)
    return f"exit {code}\n{output}"


def run_check(work_dir: Path, command: tuple[str, ...] = DEFAULT_CHECK_COMMAND) -> tuple[bool, str]:
    """Run the checking command; return whether it passed and the tail of its output."""
    code, output = run_process(work_dir, list(command))
    return code == 0, f"exit {code}\n{output}"


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_notebook",
            "description": "Read a Jupyter notebook as plain text (cells in order).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file in the work folder.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in the work folder.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run an allow-listed build or test command (make, gcc, ./binary, ...).",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]


def dispatch(work_dir: Path, name: str, args: dict) -> str:
    """Run the named tool with the given arguments and return its text result."""
    try:
        if name == "read_notebook":
            return read_notebook(work_dir, args["path"])
        if name == "read_file":
            return read_file(work_dir, args["path"])
        if name == "write_file":
            return write_file(work_dir, args["path"], args["content"])
        if name == "run_shell":
            return run_shell(work_dir, args["command"])
    except (KeyError, TypeError) as exc:
        return f"error: missing or invalid argument for {name}: {exc}"
    return f"error: unknown tool: {name}"
