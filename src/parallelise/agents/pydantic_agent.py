"""The Pydantic AI agent: the framework drives the loop, a validator guards the exit.

Besides the four shared tools it has a ``run_checks`` tool that runs the checks and records
whether they passed. An output validator refuses to accept a final answer until the last
call of ``run_checks`` succeeded, so the model cannot finish by describing a fix.
The framework is imported lazily so that the package works without it.

The framework's graph is driven one node at a time, so that the time limit can be checked
before each call to the model and so that the counts of requests and tokens survive a run
that ends in an error. The model is reached through an ``httpx2`` client, the HTTP library
that Pydantic AI now uses itself (it deprecates a client of the older ``httpx`` library), which
waits out HTTP 429 answers like the other agents.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from parallelise.agents import tools
from parallelise.agents.base import (
    AgentResult,
    AgentUnavailable,
    Deadline,
    LatencyRecorder,
    build_system_prompt,
)
from parallelise.model import throttled_async_client_class

if TYPE_CHECKING:
    from parallelise.config import Settings
    from parallelise.model import ModelClient

log = logging.getLogger("parallelise.agents.pydantic")

INSTALL_HINT = "install the framework with: pip install 'parallelise[agents]'"


@dataclass
class State:
    """Run state shared with the tools: whether the checks last passed."""

    checks_passed: bool = False


def _import_framework():
    """Import the pieces of Pydantic AI that are needed, or raise AgentUnavailable."""
    try:
        import httpx2
        import pydantic_ai
        from pydantic_ai.exceptions import UsageLimitExceeded
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        from pydantic_ai.settings import ModelSettings
        from pydantic_ai.usage import UsageLimits
    except ImportError as exc:
        raise AgentUnavailable(f"the pydantic agent needs pydantic-ai; {INSTALL_HINT}") from exc
    return pydantic_ai, {
        "RunContext": pydantic_ai.RunContext,
        "ModelRetry": pydantic_ai.ModelRetry,
        "UsageLimitExceeded": UsageLimitExceeded,
        "httpx2": httpx2,
        "OpenAIChatModel": OpenAIChatModel,
        "OpenAIProvider": OpenAIProvider,
        "ModelSettings": ModelSettings,
        "UsageLimits": UsageLimits,
    }


def build_http_client(settings: "Settings", recorder: LatencyRecorder | None = None):
    """Build the asynchronous HTTP client through which the agent reaches the model.

    It is an ``httpx2`` client, as Pydantic AI asks, that waits out HTTP 429 answers for up to
    ``settings.throttle_wait_s`` seconds per request and, when a ``recorder`` is given, times
    every request.
    """
    _, parts = _import_framework()
    client_class = throttled_async_client_class(parts["httpx2"].AsyncClient)
    return client_class(
        throttle_wait_s=settings.throttle_wait_s,
        timeout=settings.request_timeout_s,
        event_hooks=recorder.async_hooks() if recorder else None,
    )


def build_agent(
    work_dir: Path,
    settings: "Settings",
    recorder: LatencyRecorder | None = None,
    check_command: tuple[str, ...] = tools.DEFAULT_CHECK_COMMAND,
    http_client: Any = None,
):
    """Build the Pydantic AI agent for one work folder.

    The model is reached through an HTTP client at ``settings.gateway_url``: the one given as
    ``http_client`` (which the caller then closes), or else one made by ``build_http_client``.
    """
    framework, parts = _import_framework()
    work_dir = Path(work_dir)
    run_context = parts["RunContext"]
    model_retry = parts["ModelRetry"]
    if http_client is None:
        http_client = build_http_client(settings, recorder)
    provider = parts["OpenAIProvider"](
        base_url=settings.gateway_url,
        api_key=settings.api_key or "not-needed",
        http_client=http_client,
    )
    model = parts["OpenAIChatModel"](settings.model, provider=provider)
    agent = framework.Agent(
        model,
        deps_type=State,
        retries=4,
        model_settings=parts["ModelSettings"](temperature=settings.temperature),
        instructions=build_system_prompt(work_dir)
        + " After building and testing with run_shell, call run_checks. Finish only once "
        "run_checks reports success.",
    )

    @agent.tool_plain
    def read_notebook(path: str) -> str:
        """Read a Jupyter notebook as plain text (cells in order)."""
        return tools.read_notebook(work_dir, path)

    @agent.tool_plain
    def read_file(path: str) -> str:
        """Read a text file in the work folder."""
        return tools.read_file(work_dir, path)

    @agent.tool_plain
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a file in the work folder."""
        return tools.write_file(work_dir, path, content)

    @agent.tool_plain
    def run_shell(command: str) -> str:
        """Run an allow-listed build or test command (make, gcc, ./binary, ...)."""
        return tools.run_shell(work_dir, command)

    @agent.tool
    # The framework is imported lazily, so RunContext is a runtime alias here and the type
    # checker cannot see it as a class; the agent library reads the real class at run time.
    def run_checks(ctx: run_context[State]) -> str:  # type: ignore[valid-type]
        """Build and run the unit tests (the checks) and record whether they passed."""
        passed, output = tools.run_check(work_dir, check_command)
        ctx.deps.checks_passed = passed  # type: ignore[attr-defined]
        return output

    @agent.output_validator
    def must_pass(ctx: run_context[State], output: str) -> str:  # type: ignore[valid-type]
        # The exit is guarded: no final answer is accepted until run_checks is green.
        if not ctx.deps.checks_passed:  # type: ignore[attr-defined]
            raise model_retry(
                "The checks have not passed. Build the code and call run_checks; finish only "
                "once it reports success."
            )
        return output

    return agent


@dataclass
class Progress:
    """What a run has done so far, kept outside the run so that a failure cannot lose it."""

    run: Any = None
    time_limit_reached: bool = False
    budget_spent: bool = False
    error: str = ""


def usage_counts(progress: Progress) -> tuple[int, int, int]:
    """Return (requests, prompt tokens, completion tokens) for the run so far."""
    if progress.run is None:
        return 0, 0, 0
    # Older releases of the framework expose usage as a method, newer ones as a property.
    usage = progress.run.usage() if callable(progress.run.usage) else progress.run.usage
    return int(usage.requests), int(usage.input_tokens), int(usage.output_tokens)


async def drive(agent, framework, parts, task, state, step_budget, deadline, progress, http_client):
    """Run the agent's graph one node at a time, and record how it ended in ``progress``.

    Before each node that calls the model, the time limit is checked; if it has passed, the run
    stops there. A failure of the model service or the framework is recorded in ``progress``
    and not raised, so that the caller can still report the counts of what was done.
    """
    try:
        limits = parts["UsageLimits"](request_limit=step_budget)
        async with agent.iter(task, deps=state, usage_limits=limits) as agent_run:
            progress.run = agent_run
            node = agent_run.next_node
            while not framework.Agent.is_end_node(node):
                if framework.Agent.is_model_request_node(node) and deadline.expired():
                    progress.time_limit_reached = True
                    break
                node = await agent_run.next(node)
    except parts["UsageLimitExceeded"] as exc:
        log.warning("pydantic agent stopped: %s", exc)
        progress.budget_spent = True
    except Exception as exc:  # a failed model service must not hide the counts or the check
        log.warning("pydantic agent stopped: %s: %s", type(exc).__name__, exc)
        progress.error = f"{type(exc).__name__}: {exc}"
    finally:
        await http_client.aclose()


class PydanticAgent:
    """Pydantic AI with an output validator that holds the exit until the checks pass."""

    name = "pydantic"

    def run(
        self,
        work_dir: Path,
        task: str,
        settings: "Settings",
        client: "ModelClient",
        step_budget: int = 80,
    ) -> AgentResult:
        """Run the agent; ``step_budget`` limits the number of model requests.

        The run ends when the checks pass, when the step budget or the time limit in
        ``settings`` is spent, or when the model service or the framework fails; in every
        case the result holds the counts of the requests that were made.
        """
        framework, parts = _import_framework()
        recorder = LatencyRecorder()
        deadline = Deadline(settings.time_limit_s)
        http_client = build_http_client(settings, recorder)
        agent = build_agent(Path(work_dir), settings, recorder, http_client=http_client)
        state = State()
        progress = Progress()
        asyncio.run(
            drive(agent, framework, parts, task, state, step_budget, deadline, progress,
                  http_client)
        )
        requests, prompt, completion = usage_counts(progress)
        if state.checks_passed:
            notes = "the checks passed"
        elif progress.budget_spent:
            notes = "stopped: the step budget was spent without passing checks"
        else:
            notes = "ended without passing checks"
        return AgentResult(
            finished=state.checks_passed,
            steps=requests,
            requests=requests,
            prompt_tokens=prompt,
            completion_tokens=completion,
            latencies_s=list(recorder.latencies_s),
            notes=notes,
            time_limit_reached=progress.time_limit_reached,
            error=progress.error,
        )
