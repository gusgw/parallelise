"""Reference deep agent: deepagents.create_deep_agent over the local model.

Tuned for a medium (12B) local model after conversation-log analysis (see RUN_NOTES):
- a registered HarnessProfile drops the todo middleware (whose extra graph node
  made one loop iteration cost 3 recursion steps instead of 2), the dead-weight
  summarization middleware (its 170k-token trigger can never fire inside a 32k
  context), and the sub-agent `task` tool (a 6.9k-char description the small
  model doesn't need);
- the backend inherits the environment so `execute` can actually run commands;
- the system prompt agrees with the framework's own conventions (virtual paths
  start with '/') instead of contradicting them, warns that read_file's
  line-number prefixes are not file text, and names the exact test command.
"""
from __future__ import annotations

from deepagents import FilesystemMiddleware, create_deep_agent
from deepagents.backends import LocalShellBackend

from harness_common import model as hc_model
from harness_common import mistral_compat
from harness_common import prompts

from agent_langgraph.middleware import FinishGateMiddleware, RepeatGuardMiddleware

DEFAULT_TEST_CMD = [".venv/bin/python", "-m", "pytest", "-q"]

# Matches deepagents' FilesystemMiddleware conventions; do NOT reuse the shared
# prompts.STRATEGY here — its "paths are relative, never absolute" rule directly
# contradicts the middleware's "all file paths must start with a /" text.
# No backticks anywhere: a 12B copies quote characters into tool arguments.
DEEPAGENTS_SYSTEM = (
    "You are a coding agent working on a Python codebase. "
    "Use your tools to read and edit files and run tests; make minimal changes. "
    "File tool paths start with '/' ('/' is the repository root). "
    "read_file returns the file text exactly as it is — copy edit_file's "
    "old_string exactly from it. edit_file replaces one short snippet: pass a "
    "few exact lines as old_string, never a whole file. If an edit fails, the "
    "error quotes what the file really contains — copy from that. To fix a "
    "failing import or missing name, grep with pattern \"def \" and glob "
    "\"**/*.py\" to list the functions that really exist, compare them to what "
    "the test imports, then edit. Never describe a fix in text without applying "
    "it: make the edit, rerun the tests, and only finish after the tests pass."
)

# Kept for the smalltools variant, which uses our own relative-path tools.
SHORT_SYSTEM = (
    "You are a coding agent working on a Python codebase. "
    "Use your tools to read and edit files and run tests. " + prompts.STRATEGY +
    " Make minimal changes."
)

_PROFILES_REGISTERED: set[str] = set()


def _register_small_model_profile(model_name: str) -> None:
    """Register a deepagents harness profile that trims the stack for a small model."""
    if model_name in _PROFILES_REGISTERED:
        return
    from deepagents.profiles.harness.harness_profiles import (
        GeneralPurposeSubagentProfile,
        HarnessProfile,
        register_harness_profile,
    )
    from langchain.agents.middleware import TodoListMiddleware

    register_harness_profile(
        "openai:" + model_name,
        HarnessProfile(
            excluded_middleware=frozenset({TodoListMiddleware, "SummarizationMiddleware"}),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
            # replaced by our raw-text pair below: the middleware's cat -n read
            # view made the 12B copy phantom indentation into edit old_strings
            excluded_tools=frozenset({"read_file", "edit_file"}),
        ),
    )
    _PROFILES_REGISTERED.add(model_name)


def _override_tools(root: str) -> list:
    """Raw-text read_file + forgiving edit_file, honouring the '/'-path style."""
    from harness_common.tools import fs as fstools

    def _rel(file_path: str) -> str:
        return file_path.lstrip("/") or "."

    def _safe(thunk):
        try:
            return thunk()
        except Exception as exc:
            return "ERROR: " + str(exc)

    def read_file(file_path: str, start: int = 1, count: int | None = None) -> str:
        """Read a file and return its raw text exactly as it is (no line numbers). Large files are truncated; pass start and count for a line range, or grep first."""
        return _safe(lambda: fstools.read_file(root, _rel(file_path), start, count))

    def edit_file(file_path: str, old_string: str, new_string: str | None = None,
                  content: str | None = None) -> str:
        """Replace old_string with new_string (all occurrences). Copy old_string exactly from read_file output; on failure the error quotes the real file text."""
        # models blend write_file's schema into edit_file: accept content as alias
        replacement = new_string if new_string is not None else content
        if replacement is None:
            return "ERROR: pass new_string (the replacement text)."
        return _safe(lambda: fstools.edit_file(root, _rel(file_path), old_string, replacement))

    return [read_file, edit_file]


def _test_hint(test_cmd: list[str] | None) -> str:
    """One sentence naming the exact shell command that runs the target's tests."""
    cmd = " ".join(test_cmd or DEFAULT_TEST_CMD)
    return ' Run the tests with execute(command="' + cmd + '").'


def build_model():
    """Build a LangChain chat model pointed at the local OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI

    cfg = hc_model.model_config()
    return ChatOpenAI(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"] or "not-needed",
        model=cfg["model"],
        temperature=hc_model.default_temperature(),
        max_tokens=3000,   # a 12B will otherwise emit multi-thousand-token essays
        # relabel tool-call ids to 9-char (Mistral template); safe for other servers
        http_client=mistral_compat.make_sync_client(),
        http_async_client=mistral_compat.make_async_client(),
    )


def build_agent(root: str, *, test_cmd: list[str] | None = None,
                system_prompt: str = DEEPAGENTS_SYSTEM):
    """Create the tuned full deep agent (trimmed profile) rooted at the target repo."""
    cfg = hc_model.model_config()
    _register_small_model_profile(cfg["model"])
    return create_deep_agent(
        model=build_model(),
        tools=_override_tools(root),
        system_prompt=system_prompt + _test_hint(test_cmd),
        backend=LocalShellBackend(root_dir=root, virtual_mode=True, inherit_env=True),
        middleware=[RepeatGuardMiddleware(), FinishGateMiddleware()],
    )


def build_minimal_agent(root: str, *, test_cmd: list[str] | None = None,
                        system_prompt: str = DEEPAGENTS_SYSTEM):
    """Trimmed agent: LangChain create_agent + only deepagents' filesystem tools.

    Drops all of deepagents' other middleware, to isolate the contribution of
    the filesystem middleware (and its prompt) from the rest of the stack.
    """
    from langchain.agents import create_agent

    backend = LocalShellBackend(root_dir=root, virtual_mode=True, inherit_env=True)
    return create_agent(
        model=build_model(),
        system_prompt=system_prompt + _test_hint(test_cmd),
        middleware=[FilesystemMiddleware(backend=backend), RepeatGuardMiddleware(),
                    FinishGateMiddleware()],
    )


def build_smalltools_agent(root: str, *, test_cmd: list[str] | None = None,
                           system_prompt: str = SHORT_SYSTEM):
    """Smallest LangGraph agent: create_agent + agent_min's six tools, no middleware.

    Mirrors agent_min's surface but through LangGraph, to isolate what the
    framework itself adds or costs versus our hand-rolled loop.
    """
    from langchain.agents import create_agent

    from harness_common.tools import fs as fstools
    from harness_common.tools import run_tests as testrun

    def _safe(thunk):
        try:
            return thunk()
        except Exception as exc:
            return "ERROR: " + str(exc)

    def read_file(path: str, start: int = 1, count: int | None = None) -> str:
        """Read a text file. Large files are truncated; pass start+count for a line range, or grep first."""
        return _safe(lambda: fstools.read_file(root, path, start, count))

    def ls(path: str = ".") -> str:
        """List a directory in the repo."""
        return _safe(lambda: "\n".join(fstools.ls(root, path)))

    def grep(pattern: str, path: str = ".") -> str:
        """Search the repo for a regular expression."""
        return _safe(lambda: "\n".join(fstools.grep(root, pattern, path)) or "(no matches)")

    def edit_file(path: str, old: str, new: str) -> str:
        """Replace exact text in a repo file."""
        return _safe(lambda: fstools.edit_file(root, path, old, new))

    def write_file(path: str, content: str) -> str:
        """Create or overwrite a repo file."""
        return _safe(lambda: fstools.write_file(root, path, content))

    def run_tests() -> str:
        """Run the project's test suite."""
        return _safe(lambda: testrun.summarize(testrun.run_tests(root, test_cmd)))

    return create_agent(
        model=build_model(),
        system_prompt=system_prompt,
        tools=[read_file, ls, grep, edit_file, write_file, run_tests],
        middleware=[RepeatGuardMiddleware(), FinishGateMiddleware()],
    )
