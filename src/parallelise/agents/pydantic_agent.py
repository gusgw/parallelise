"""The Pydantic AI agent: the framework drives the loop, a validator guards the exit.

Besides the four shared tools it has a ``check_gate`` tool that runs the checks and records
whether they passed. An output validator refuses to accept a final answer until the last
call of ``check_gate`` succeeded, so the model cannot finish by describing a fix.
The framework is imported lazily so that the package works without it.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from parallelise.agents import tools
from parallelise.agents.base import (
    AgentResult,
    AgentUnavailable,
    LatencyRecorder,
    build_system_prompt,
)

if TYPE_CHECKING:
    from parallelise.config import Settings
    from parallelise.model import ModelClient

log = logging.getLogger("parallelise.agents.pydantic")

INSTALL_HINT = "install the framework with: pip install 'parallelise[agents]'"


@dataclass
class State:
    """Run state shared with the tools: whether the checks last passed."""

    gate_passed: bool = False


def _import_framework():
    """Import the pieces of Pydantic AI that are needed, or raise AgentUnavailable."""
    try:
        import pydantic_ai
        from pydantic_ai.exceptions import (
            ModelHTTPError,
            UnexpectedModelBehavior,
            UsageLimitExceeded,
        )
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        from pydantic_ai.settings import ModelSettings
        from pydantic_ai.usage import UsageLimits
    except ImportError as exc:
        raise AgentUnavailable(f"the pydantic agent needs pydantic-ai; {INSTALL_HINT}") from exc
    return pydantic_ai, {
        "RunContext": pydantic_ai.RunContext,
        "ModelRetry": pydantic_ai.ModelRetry,
        "errors": (ModelHTTPError, UnexpectedModelBehavior, UsageLimitExceeded),
        "OpenAIChatModel": OpenAIChatModel,
        "OpenAIProvider": OpenAIProvider,
        "ModelSettings": ModelSettings,
        "UsageLimits": UsageLimits,
    }


def build_agent(
    work_dir: Path,
    settings: "Settings",
    recorder: LatencyRecorder | None = None,
    gate_command: tuple[str, ...] = tools.DEFAULT_GATE_COMMAND,
):
    """Build the Pydantic AI agent for one work folder.

    The model is reached through its own HTTP client at ``settings.gateway_url``. When a
    ``recorder`` is given, it times every request.
    """
    import httpx

    framework, parts = _import_framework()
    work_dir = Path(work_dir)
    run_context = parts["RunContext"]
    model_retry = parts["ModelRetry"]
    http_client = httpx.AsyncClient(
        timeout=settings.request_timeout_s,
        event_hooks=recorder.async_hooks() if recorder else None,
    )
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
        + " After building and testing with run_shell, call check_gate. Finish only once "
        "check_gate reports success.",
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
    def check_gate(ctx: run_context[State]) -> str:
        """Build and run the unit tests (the checks) and record whether they passed."""
        passed, output = tools.run_gate(work_dir, gate_command)
        ctx.deps.gate_passed = passed
        return output

    @agent.output_validator
    def must_pass(ctx: run_context[State], output: str) -> str:
        # The exit is guarded: no final answer is accepted until check_gate is green.
        if not ctx.deps.gate_passed:
            raise model_retry(
                "The checks have not passed. Build the code and call check_gate; finish only "
                "once it reports success."
            )
        return output

    return agent


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
        """Run the agent; ``step_budget`` limits the number of model requests."""
        _, parts = _import_framework()
        recorder = LatencyRecorder()
        agent = build_agent(Path(work_dir), settings, recorder)
        state = State()
        try:
            outcome = agent.run_sync(
                task,
                deps=state,
                usage_limits=parts["UsageLimits"](request_limit=step_budget),
            )
        except parts["errors"] as exc:
            # The model did not produce a passing conversion within budget. Report a clean
            # failure; whatever files it wrote are still on disk for grading.
            log.warning("pydantic agent stopped: %s: %s", type(exc).__name__, exc)
            return AgentResult(
                False, 0, 0, 0, 0, list(recorder.latencies_s), f"{type(exc).__name__}: {exc}"
            )
        # Older releases of the framework expose usage as a method, newer ones as a property.
        usage = outcome.usage() if callable(outcome.usage) else outcome.usage
        return AgentResult(
            finished=state.gate_passed,
            steps=usage.requests,
            requests=usage.requests,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            latencies_s=list(recorder.latencies_s),
            notes="the checks passed" if state.gate_passed else "ended without passing checks",
        )
