"""The three agent designs compared in this project, and a lookup by name."""

from __future__ import annotations

from parallelise.agents.base import Agent, AgentResult
from parallelise.errors import ParalleliseError

__all__ = ["Agent", "AgentResult", "AGENT_NAMES", "get_agent"]

AGENT_NAMES = ("basic", "pydantic", "langgraph")


def get_agent(name: str) -> Agent:
    """Return an agent by name. The framework modules are imported only when asked for."""
    if name == "basic":
        from parallelise.agents.basic import BasicAgent

        return BasicAgent()
    if name == "pydantic":
        from parallelise.agents.pydantic_agent import PydanticAgent

        return PydanticAgent()
    if name == "langgraph":
        from parallelise.agents.langgraph_agent import LangGraphAgent

        return LangGraphAgent()
    raise ParalleliseError(f"The agent {name!r} is not one of {', '.join(AGENT_NAMES)}.")
