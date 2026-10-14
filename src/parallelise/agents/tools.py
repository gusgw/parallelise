"""The tool surface shared by all three agent designs.

Every function returns text and never raises: a failure comes back as a string beginning
"error:", so that the model can read it and carry on. A missing optional file, for example,
must not end the run. Paths are confined to the work folder.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path

log = logging.getLogger("parallelise.agents.tools")

# Programs the model may start. The command is run as an argument list with no shell, so
# ';', '|' and '&&' are passed as plain arguments and cannot chain further commands.
ALLOWED_COMMANDS = frozenset(
    {"make", "gcc", "cc", "clang", "mpicc", "mpirun", "nvcc", "bash", "sh", "ls", "cat", "python3"}
)
COMMAND_TIMEOUT_S = 600
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


def run_process(
    work_dir: Path, parts: list[str], timeout_s: int = COMMAND_TIMEOUT_S
) -> tuple[int, str]:
    """Run an argument list in the work folder; return (exit code, tail of the output).

    An exit code of -1 means the program could not be started or ran out of time.
    """
    try:
        proc = subprocess.run(
            parts, cwd=work_dir, capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired:
        return -1, f"error: the command ran for more than {timeout_s} seconds and was stopped"
    except OSError as exc:
        return -1, _error(exc)
    return proc.returncode, (proc.stdout + proc.stderr)[-OUTPUT_TAIL_CHARS:]


def run_shell(work_dir: Path, command: str) -> str:
    """Run an allow-listed build or test command in the work folder and return its output."""
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return _error(exc)
    if not parts:
        return "error: empty command"
    program = Path(parts[0]).name
    if program not in ALLOWED_COMMANDS and not parts[0].startswith("./"):
        return f"error: command not allowed: {parts[0]}"
    if program in {"bash", "sh"} and "-c" in parts[1:]:
        return "error: running a command string through bash or sh is not allowed"
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
