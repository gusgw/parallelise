"""Challenger agent: Pydantic AI with agent_min's tool surface over harness_common.

Three variants (selected by AGENT_PYDANTIC_VARIANT; see __main__) ablate Pydantic
AI's "don't finish prematurely" levers on a 12B model:

- `build_agent` (full): deps-tracked verification + an output validator that
  refuses to finish (re-asks the model) until the tests actually PASS. Forcing via
  the validator only — per-step tool_choice="required" made the 12B emit over-long
  forced tool calls that overran max_tokens and crashed the run, so it was dropped.
- `build_min_agent` (min): the same six tools and ModelRetry-based error feedback,
  but NO forcing at all — the closest analogue to agent_min, to isolate whether the
  framework alone helps.
- `build_typed_agent` (typed): forces the run to END via a structured ToolOutput
  (FixReport) instead of free prose, but does NOT verify the report. It scored zero
  on both models: the 12B ticked tests_pass=True over a still-broken suite, and the
  31B, pushed to answer through the report tool, returned empty/unusable finals,
  exhausted its output-retry budget, and crashed. Kept as-is for the record.
- `build_typed_checked_agent` (typed_checked): the same structured finish, but the
  report is only accepted once the tests genuinely pass (deps + output validator,
  as in full) and with a larger output-retry budget so an awkward finish degrades
  rather than crashes. Isolates whether a *verified* structured finish works.

All variants: tool errors raise ModelRetry (so the retry loop engages), and exact
repeats / oversized edit snippets are rejected the same way.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext, ToolOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

from harness_common import mistral_compat
from harness_common import model as hc_model
from harness_common import prompts
from harness_common.tools import fs as fstools
from harness_common.tools import run_tests as testrun

SHORT_SYSTEM = (
    "You can read and edit repository files and run tests using the provided "
    "tools. Call a tool to get real data; never invent file contents. " +
    prompts.STRATEGY + " Run run_tests to verify your fix; only then reply "
    "with a short summary."
)

MAX_EDIT_SNIPPET = 4000   # chars; edit_file old/new must be short unique snippets

REPEAT_MSG = (
    "You already made this exact call and saw its result; repeating it returns "
    "the same thing. Take a different action."
)


@dataclass
class RunState:
    """Per-run verification state."""
    tests_run: bool = False       # run_tests called since the last edit
    tests_passed: bool = False    # ...and it reported success
    last_summary: str = ""        # the most recent run_tests summary (for feedback)


class FixReport(BaseModel):
    """Structured finish for the `typed` variant."""
    summary: str
    tests_pass: bool


def build_model() -> OpenAIChatModel:
    """Build a Pydantic AI model pointed at the local OpenAI-compatible endpoint."""
    cfg = hc_model.model_config()
    provider = OpenAIProvider(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"] or "not-needed",
        http_client=mistral_compat.make_async_client(),
    )
    return OpenAIChatModel(cfg["model"], provider=provider)


def _plain_settings() -> ModelSettings:
    return ModelSettings(temperature=hc_model.default_temperature(),
                         parallel_tool_calls=False, max_tokens=6000)


def check_snippet_size(old: str, new: str) -> None:
    """Reject whole-file edit arguments; they blow up the resent history."""
    if len(old) > MAX_EDIT_SNIPPET or len(new) > MAX_EDIT_SNIPPET:
        raise ModelRetry("old/new are far too large. Pass a short unique snippet "
                         "of a few exact lines, not a whole file.")


def _safe(thunk):
    """Run a tool thunk; raise ModelRetry so the framework retries with guidance."""
    try:
        return thunk()
    except ModelRetry:
        raise
    except Exception as exc:
        raise ModelRetry(str(exc)) from exc


def _make_no_repeat():
    """A per-agent guard: returns a nudge string on an exact-repeat call, else None.

    Deliberately a returned tool result, not a ModelRetry — a repeat is not a tool
    *failure*, and counting it against the tool-retry budget crashed the run.
    """
    last: dict[str, str | None] = {"sig": None}

    def no_repeat(name: str, args: dict) -> str | None:
        sig = name + ":" + json.dumps(args, sort_keys=True, default=str)
        repeated = sig == last["sig"]
        last["sig"] = sig
        return REPEAT_MSG if repeated else None

    return no_repeat


def _attach_readonly_tools(agent, root, no_repeat):
    """read_file / ls / grep — shared, side-effect-free, no deps."""

    @agent.tool_plain
    def read_file(path: str, start: int = 1, count: int | None = None) -> str:
        """Read a text file. Large files are truncated; pass start+count for a line range, or grep first."""
        return no_repeat("read_file", {"path": path, "start": start, "count": count}) \
            or _safe(lambda: fstools.read_file(root, path, start, count))

    @agent.tool_plain
    def ls(path: str = ".") -> str:
        """List a directory in the repo."""
        return no_repeat("ls", {"path": path}) \
            or _safe(lambda: "\n".join(fstools.ls(root, path)))

    @agent.tool_plain
    def grep(pattern: str, path: str = ".") -> str:
        """Search the repo for a regular expression."""
        return no_repeat("grep", {"pattern": pattern, "path": path}) \
            or _safe(lambda: "\n".join(fstools.grep(root, pattern, path)) or "(no matches)")


def _attach_plain_write_tools(agent, root, test_cmd, no_repeat):
    """edit_file / write_file / run_tests with no verification tracking (min, typed)."""

    @agent.tool_plain
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace exact text in a repo file. old and new are short snippets, never a whole file."""
        repeated = no_repeat("edit_file", {"path": path, "old": old, "new": new})
        if repeated:
            return repeated
        check_snippet_size(old, new)
        return _safe(lambda: fstools.edit_file(root, path, old, new))

    @agent.tool_plain
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a repo file."""
        return _safe(lambda: fstools.write_file(root, path, content))

    @agent.tool_plain
    def run_tests() -> str:
        """Run the project's test suite."""
        return _safe(lambda: testrun.summarize(testrun.run_tests(root, test_cmd)))


def _attach_deps_write_tools(agent, root, test_cmd, no_repeat):
    """edit_file / write_file / run_tests that record real test state on the deps.

    Shared by the variants that gate the finish on reality (full, typed_checked):
    any edit clears the verified flag; run_tests sets tests_passed from the real
    result, so a downstream output validator can refuse to finish until it is True.
    """

    @agent.tool
    def edit_file(ctx: RunContext[RunState], path: str, old: str, new: str) -> str:
        """Replace exact text in a repo file. old and new are short snippets, never a whole file."""
        repeated = no_repeat("edit_file", {"path": path, "old": old, "new": new})
        if repeated:
            return repeated
        check_snippet_size(old, new)
        ctx.deps.tests_run = False        # any edit invalidates prior verification
        ctx.deps.tests_passed = False
        return _safe(lambda: fstools.edit_file(root, path, old, new))

    @agent.tool
    def write_file(ctx: RunContext[RunState], path: str, content: str) -> str:
        """Create or overwrite a repo file."""
        ctx.deps.tests_run = False
        ctx.deps.tests_passed = False
        return _safe(lambda: fstools.write_file(root, path, content))

    @agent.tool
    def run_tests(ctx: RunContext[RunState]) -> str:
        """Run the project's test suite."""
        result = _safe(lambda: testrun.run_tests(root, test_cmd))
        ctx.deps.tests_run = True
        ctx.deps.tests_passed = bool(result.ok)
        summary = testrun.summarize(result)
        ctx.deps.last_summary = summary
        return summary


def build_min_agent(root: str, *, test_cmd: list[str] | None = None,
                    instructions: str = SHORT_SYSTEM) -> Agent:
    """Six tools + ModelRetry feedback, NO forcing — the framework-only baseline."""
    agent = Agent(build_model(), instructions=instructions,
                  retries={"tools": 5, "output": 4}, model_settings=_plain_settings())
    no_repeat = _make_no_repeat()
    _attach_readonly_tools(agent, root, no_repeat)
    _attach_plain_write_tools(agent, root, test_cmd, no_repeat)
    return agent


def build_typed_agent(root: str, *, test_cmd: list[str] | None = None,
                      instructions: str = SHORT_SYSTEM) -> Agent:
    """Six tools, but the run must FINISH via a structured FixReport (no prose path)."""
    agent = Agent(build_model(), instructions=instructions,
                  output_type=ToolOutput(FixReport, name="report_fix"),
                  retries={"tools": 5, "output": 4}, model_settings=_plain_settings())
    no_repeat = _make_no_repeat()
    _attach_readonly_tools(agent, root, no_repeat)
    _attach_plain_write_tools(agent, root, test_cmd, no_repeat)
    return agent


def build_typed_checked_agent(root: str, *, test_cmd: list[str] | None = None,
                              instructions: str = SHORT_SYSTEM) -> Agent:
    """typed, but the structured FixReport is only accepted once tests really pass.

    The plain `typed` variant scored zero on both models for two distinct reasons,
    both flowing from making the structured report the only way to finish:
    - the 12B simply ticked tests_pass=True while the suite was still broken;
      nothing checked the claim.
    - the 31B, forced to answer through the report tool, kept returning empty or
      unusable final messages, exhausted its output-retry budget, and the run died.
    This version verifies the claim (deps + an output validator, as in `full`) and
    gives a larger output-retry budget so an awkward finish degrades instead of
    crashing. It isolates one thing: does a *verified* structured finish work?
    """
    # Kept as a known-zero control. The output-retry budget is a cumulative whole-run
    # counter, and on this gemma4 build the structured-output TOOL provokes a storm of
    # empty/non-actionable turns that exhaust it before a multi-edit fix completes. No
    # budget rescues it: raising output retries to 80 just trades exhaustion for a
    # wall-clock timeout (2x each at n=4), and the tool-free PromptedOutput mode fares
    # no better (the model stops calling tools entirely and only re-emits the report).
    # A moderate budget therefore just fails fast and cheap. The lesson: the value is
    # the verification gate, not the structured finish — deliver it via prose (build_agent).
    agent = Agent(build_model(), deps_type=RunState, instructions=instructions,
                  output_type=ToolOutput(FixReport, name="report_fix"),
                  retries={"tools": 5, "output": 12}, model_settings=_plain_settings())
    no_repeat = _make_no_repeat()
    _attach_readonly_tools(agent, root, no_repeat)
    _attach_deps_write_tools(agent, root, test_cmd, no_repeat)

    @agent.output_validator
    def must_really_pass(ctx: RunContext[RunState], output: FixReport) -> FixReport:
        if ctx.deps.tests_passed:
            output.tests_pass = True            # the claim now matches reality
            return output
        if not ctx.deps.tests_run:
            raise ModelRetry("Do not report yet. Call run_tests first, then report.")
        # Concrete feedback + a directive back to editing: left to its own devices the
        # model re-submits the report instead of fixing, so spell out the next action
        # and forbid the shortcut of weakening the tests to make errors disappear.
        raise ModelRetry(
            "The tests still fail, so report_fix is rejected. Do NOT call report_fix "
            "again yet, and do NOT delete tests or replace their bodies with pass to "
            "silence the error. Read the real code to find the correct names/behaviour, "
            "make another edit that fixes the actual cause, then call run_tests. Only "
            "call report_fix after run_tests reports success.\nLatest run_tests result:\n"
            + (ctx.deps.last_summary or "(none)"))

    return agent


def build_agent(root: str, *, test_cmd: list[str] | None = None,
                instructions: str = SHORT_SYSTEM) -> Agent:
    """Full: verify-tracked deps + a must-pass output validator (re-ask forcing)."""
    agent = Agent(build_model(), deps_type=RunState, instructions=instructions,
                  retries={"tools": 5, "output": 4}, model_settings=_plain_settings())
    no_repeat = _make_no_repeat()
    _attach_readonly_tools(agent, root, no_repeat)
    _attach_deps_write_tools(agent, root, test_cmd, no_repeat)

    @agent.output_validator
    def must_pass(ctx: RunContext[RunState], output: str) -> str:
        if ctx.deps.tests_passed:
            return output
        if not ctx.deps.tests_run:
            raise ModelRetry("Do not summarize yet. Call run_tests first.")
        raise ModelRetry("The tests are not passing yet. Keep editing and rerun "
                         "run_tests; only finish once it reports success.")

    return agent
