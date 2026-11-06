"""The Pydantic AI agent: the framework drives the loop, a validator guards the exit.

The tools are the shared ones, built from their schemas. An output validator runs the shared
check whenever the model gives a final answer, and refuses the answer until the check passes,
so the model cannot finish by describing a fix. It may refuse as many times as the step budget
allows, so that this design has the same budget as the others.

The framework's graph is driven one node at a time, so that the time limit can be checked
before each call to the model. The model is reached through an ``httpx2`` client, the HTTP
library that Pydantic AI uses, whose transport is the run's meter.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from parallelise.agents.base import AgentResult, Toolset
from parallelise.errors import AgentUnavailable

if TYPE_CHECKING:
    from parallelise.config import Settings
    from parallelise.model import ModelClient

log = logging.getLogger("parallelise.agents.pydantic")

INSTALL_HINT = "install the framework with: pip install 'parallelise[agents]'"


def build_agent(toolset: Toolset, system_prompt: str, settings: Settings, http_client: Any,
                step_budget: int) -> tuple[Any, dict[str, bool]]:
    """Build the Pydantic AI agent with the shared tools; return it and its exit's state."""
    try:
        import pydantic_ai
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        from pydantic_ai.settings import ModelSettings
    except ImportError as exc:
        raise AgentUnavailable(f"the pydantic agent needs pydantic-ai; {INSTALL_HINT}") from exc
    provider = OpenAIProvider(base_url=settings.gateway_url,
                              api_key=settings.api_key or "not-needed", http_client=http_client)
    model = OpenAIChatModel(settings.model, provider=provider)

    def caller(name: str):
        return lambda **arguments: toolset.run(name, arguments)

    tools = [pydantic_ai.Tool.from_schema(caller(s["function"]["name"]), s["function"]["name"],
                                          s["function"].get("description"),
                                          s["function"]["parameters"])
             for s in toolset.schemas]  # fmt: skip
    agent = pydantic_ai.Agent(model, retries=step_budget, instructions=system_prompt, tools=tools,
                              model_settings=ModelSettings(temperature=settings.temperature))
    exit_ = {"passed": False}

    @agent.output_validator
    def must_pass(output: str) -> str:
        # The exit is guarded: no final answer is accepted until the shared check passes.
        passed, feedback = toolset.finish()
        if not passed:
            raise pydantic_ai.ModelRetry(feedback)
        exit_["passed"] = True
        return output

    return agent, exit_


async def _drive(agent: Any, task: str, step_budget: int, deadline: float,
                 outcome: dict[str, Any]) -> None:
    """Run the agent's graph one node at a time, and record how it ended in ``outcome``.

    Before each node that calls the model, the time limit is checked. A failure of the model
    service or the framework is recorded, not raised, so that the run is still judged.
    """
    import pydantic_ai
    from pydantic_ai.exceptions import UsageLimitExceeded
    from pydantic_ai.usage import UsageLimits

    try:
        async with agent.iter(task, usage_limits=UsageLimits(request_limit=step_budget)) as run:
            node = run.next_node
            while not pydantic_ai.Agent.is_end_node(node):
                if pydantic_ai.Agent.is_model_request_node(node) and time.monotonic() >= deadline:
                    outcome["time_limit_reached"] = True
                    return
                node = await run.next(node)
    except UsageLimitExceeded as exc:
        log.warning("pydantic agent stopped: %s", exc)
        outcome["notes"] = "stopped: the step budget was spent without passing checks"
    except Exception as exc:  # a failed model service must not stop the judgement
        log.warning("pydantic agent stopped: %s: %s", type(exc).__name__, exc)
        outcome["error"] = f"{type(exc).__name__}: {exc}"


class PydanticAgent:
    """Pydantic AI with an output validator that holds the exit until the check passes."""

    def run(self, work_dir: Path, task: str, settings: Settings, client: ModelClient,
            step_budget: int, toolset: Toolset, system_prompt: str) -> AgentResult:
        """Run the agent; ``step_budget`` limits the number of model requests."""
        try:
            import httpx2
        except ImportError as exc:
            raise AgentUnavailable(f"the pydantic agent needs httpx2; {INSTALL_HINT}") from exc
        http_client = httpx2.AsyncClient(timeout=settings.request_timeout_s,
                                         transport=client.meter.async_transport(httpx2))
        agent, exit_ = build_agent(toolset, system_prompt, settings, http_client, step_budget)
        outcome: dict[str, Any] = {"notes": "ended without passing checks"}

        async def go() -> None:
            try:
                await _drive(agent, task, step_budget,
                             time.monotonic() + settings.time_limit_s, outcome)  # fmt: skip
            finally:
                await http_client.aclose()

        asyncio.run(go())
        return AgentResult(exit_["passed"],
                           "the checks passed" if exit_["passed"] else outcome["notes"],
                           outcome.get("time_limit_reached", False), outcome.get("error", ""))
