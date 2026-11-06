"""Shared definitions for the three agent designs.

Every design receives the same prompt and the same toolset from its caller, so that the
comparison between them is fair, and reaches the model through the caller's client, whose meter
counts every call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from parallelise.config import Settings
    from parallelise.model import ModelClient


@dataclass
class AgentResult:
    """How an agent's run ended.

    ``finished`` is true only when the check passed at the end. ``time_limit_reached`` is true
    when the run was stopped, between steps, because the time limit had passed, and ``error``
    describes a failure of the model or the framework that ended it.
    """

    finished: bool
    notes: str = ""
    time_limit_reached: bool = False
    error: str = ""


@dataclass
class Toolset:
    """What an agent may do, and the check that decides when its work is finished.

    ``schemas`` describe the tools in the form the chat interface expects; ``run`` carries out
    one call, given the tool's name and arguments, and returns what the model is told; and
    ``finish`` runs whenever the model says it has finished, and returns whether the work
    passed and, if it did not, what to tell the model.
    """

    schemas: list[dict]
    run: Callable[[str, dict], str]
    finish: Callable[[], tuple[bool, str]]


def file_toolset(folder: Path, readable: list[str], writable: str, check_words: str,
                 check: Callable[[], str], finish: Callable[[], tuple[bool, str]]) -> Toolset:
    """Tools to read some files in ``folder``, to write one, and to check it.

    ``check`` returns what the model is told when it calls the tool check, and ``finish``
    decides whether the work passes when the model says it has finished.
    """
    readable = [*readable, writable]
    path_only = {"type": "object", "properties": {"path": {"type": "string"}},
                 "required": ["path"]}  # fmt: skip
    path_and_content = {"type": "object", "properties": {"path": {"type": "string"},
                                                         "content": {"type": "string"}},
                        "required": ["path", "content"]}  # fmt: skip
    schemas = [
        {"type": "function", "function": {
            "name": "read_file", "description": f"Read one of these files: {', '.join(readable)}.",
            "parameters": path_only}},
        {"type": "function", "function": {
            "name": "write_file", "description": f"Write the whole of {writable}.",
            "parameters": path_and_content}},
        {"type": "function", "function": {
            "name": "check", "description": check_words,
            "parameters": {"type": "object", "properties": {}}}},
    ]  # fmt: skip

    def run(tool: str, args: dict) -> str:
        path = str(args.get("path", "")).removeprefix("./")
        if tool == "read_file":
            if path not in readable or not (folder / path).is_file():
                return f"error: you may read only {', '.join(readable)}"
            return (folder / path).read_text(encoding="utf-8")
        if tool == "write_file":
            if path != writable:
                return f"error: you may write only {writable}"
            (folder / writable).write_text(str(args.get("content", "")), encoding="utf-8")
            return f"wrote {writable}"
        if tool == "check":
            if not (folder / writable).is_file():
                return f"check failed: {writable} has not been written"
            return check()
        return f"error: there is no tool named {tool!r}"

    def done() -> tuple[bool, str]:
        if not (folder / writable).is_file():
            return False, f"You have not written {writable}. Write it, then call check."
        return finish()

    return Toolset(schemas, run, done)


class Agent(Protocol):
    """The interface every agent design implements."""

    def run(self, work_dir: Path, task: str, settings: Settings, client: ModelClient,
            step_budget: int, toolset: Toolset, system_prompt: str) -> AgentResult:
        """Work in ``work_dir`` on ``task``, with ``toolset``, until its check passes or the
        budget is spent."""
        ...
