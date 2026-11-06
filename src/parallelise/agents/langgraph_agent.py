"""The LangGraph agent: a tool-calling graph inside a re-check loop.

LangChain's ``create_agent`` builds a graph that alternates between the model and the shared
tools until the model stops asking for tools. That alone would let the model finish by
describing a fix, so each pass of the graph is followed by the shared check; if it fails, the
conversation continues with the failure appended, until the step budget is spent.

The graph is streamed state by state, which lets the agent stop between steps when the time
limit has passed, and keeps the messages when a run ends in an error. The model is reached
through HTTP clients whose transports are the run's meter; the OpenAI library's own retries are
off, so that they do not repeat the meter's waiting.
"""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from parallelise.agents.base import AgentResult, Toolset
from parallelise.errors import AgentUnavailable

if TYPE_CHECKING:
    from parallelise.config import Settings
    from parallelise.model import ModelClient

log = logging.getLogger("parallelise.agents.langgraph")

INSTALL_HINT = "install the frameworks with: pip install 'parallelise[agents]'"


def build_agent(toolset: Toolset, system_prompt: str, settings: Settings,
                client: ModelClient) -> Any:
    """Build the tool-calling graph with the shared tools, reaching the model through ``client``'s
    meter."""
    try:
        from langchain.agents import create_agent
        from langchain_core.tools import StructuredTool
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr
    except ImportError as exc:
        raise AgentUnavailable(f"the langgraph agent needs langchain; {INSTALL_HINT}") from exc
    model = ChatOpenAI(
        base_url=settings.gateway_url, api_key=SecretStr(settings.api_key or "not-needed"),
        model=settings.model, temperature=settings.temperature, max_retries=0,
        http_client=httpx.Client(timeout=settings.request_timeout_s,
                                 transport=client.meter.transport()),
        http_async_client=httpx.AsyncClient(timeout=settings.request_timeout_s,
                                            transport=client.meter.async_transport()),
    )  # fmt: skip

    def caller(name: str):
        return lambda **arguments: toolset.run(name, arguments)

    tools = [StructuredTool.from_function(func=caller(s["function"]["name"]),
                                          name=s["function"]["name"],
                                          description=s["function"].get("description", ""),
                                          args_schema=s["function"]["parameters"])
             for s in toolset.schemas]  # fmt: skip
    return create_agent(model=model, system_prompt=system_prompt, tools=tools)


def _replies(messages: list) -> int:
    """The number of the model's replies so far, which is the number of requests made."""
    return sum(getattr(m, "type", "") == "ai" for m in messages)


def _model_is_called_next(state: dict) -> bool:
    """Whether the graph, after reaching ``state``, goes on to call the model.

    It does unless the last message is the model's answer with no tool calls, which ends the
    pass; the time limit stops only a model call, so a pass that is ending is not stopped.
    """
    messages = state.get("messages") or []
    if not messages or getattr(messages[-1], "type", "") != "ai":
        return True
    return bool(getattr(messages[-1], "tool_calls", None))


class LangGraphAgent:
    """LangChain's tool-calling graph with a loop that re-runs the shared check."""

    def run(self, work_dir: Path, task: str, settings: Settings, client: ModelClient,
            step_budget: int, toolset: Toolset, system_prompt: str) -> AgentResult:
        """Run passes of the graph until the check passes or ``step_budget`` requests are made."""
        try:
            from langgraph.errors import GraphRecursionError
        except ImportError as exc:
            raise AgentUnavailable(f"the langgraph agent needs langgraph; {INSTALL_HINT}") from exc
        deadline = time.monotonic() + settings.time_limit_s
        agent = build_agent(toolset, system_prompt, settings, client)
        messages: list = [{"role": "user", "content": task}]
        while _replies(messages) < step_budget:
            left = step_budget - _replies(messages)
            state: dict = {"messages": messages}
            try:
                # One request takes two steps of the graph: the model's, then the tools'.
                stream = agent.stream({"messages": messages},
                                      config={"recursion_limit": 2 * left + 1},
                                      stream_mode="values")  # fmt: skip
                with contextlib.closing(stream):
                    for state in stream:  # noqa: B007 - the last state is used below
                        if time.monotonic() >= deadline and _model_is_called_next(state):
                            log.warning("the time limit has passed; stopping the agent")
                            return AgentResult(False, time_limit_reached=True)
            except GraphRecursionError:
                break
            except Exception as exc:  # a failed model service must not stop the judgement
                log.warning("langgraph agent stopped: %s: %s", type(exc).__name__, exc)
                return AgentResult(False, error=f"{type(exc).__name__}: {exc}")
            messages = list(state["messages"])
            passed, feedback = toolset.finish()
            if passed:
                return AgentResult(True, "the checks passed")
            log.info("the checks fail; asking again")
            messages.append({"role": "user", "content": feedback})
        return AgentResult(False, "stopped: the step budget was spent without passing checks")
