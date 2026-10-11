"""Build the agent's tool schemas and bound handlers over a repo root."""
from __future__ import annotations

from typing import Any, Callable

from harness_common.tools import fs, run_tests


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """Assemble one OpenAI-style function tool schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def tool_schemas() -> list[dict[str, Any]]:
    """Return the small, typed tool surface exposed to the model."""
    s = {"type": "string"}
    i = {"type": "integer"}
    return [
        _schema("read_file",
                "Read a text file. Large files are truncated; pass start+count to "
                "read a line range, or grep first to locate code.",
                {"path": s, "start": i, "count": i}, ["path"]),
        _schema("ls", "List a directory in the repo.", {"path": s}, []),
        _schema("grep", "Search the repo for a regular expression.",
                {"pattern": s, "path": s}, ["pattern"]),
        _schema("edit_file", "Replace exact text in a repo file.",
                {"path": s, "old": s, "new": s}, ["path", "old", "new"]),
        _schema("write_file", "Create or overwrite a repo file.",
                {"path": s, "content": s}, ["path", "content"]),
        _schema("run_tests", "Run the project's test suite.", {}, []),
    ]


def build_handlers(root: str, test_cmd: list[str] | None = None) -> dict[str, Callable]:
    """Return name -> callable(**args) handlers bound to the repo root."""
    return {
        "read_file": lambda path, start=1, count=None: fs.read_file(root, path, start, count),
        "ls": lambda path=".": "\n".join(fs.ls(root, path)),
        "grep": lambda pattern, path=".": "\n".join(fs.grep(root, pattern, path)) or "(no matches)",
        "edit_file": lambda path, old, new: fs.edit_file(root, path, old, new),
        "write_file": lambda path, content: fs.write_file(root, path, content),
        "run_tests": lambda: run_tests.summarize(run_tests.run_tests(root, test_cmd)),
    }
