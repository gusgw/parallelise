"""The three agent designs compared in this project, and a lookup by name."""

from __future__ import annotations

from parallelise.agents.base import Agent, AgentResult, AgentUnavailable
from parallelise.errors import InputError

__all__ = ["Agent", "AgentResult", "AgentUnavailable", "AGENT_NAMES", "get_agent"]

AGENT_NAMES = ("basic", "pydantic", "langgraph")


def get_agent(name: str) -> Agent:
    """Return an agent by name: ``basic``, ``pydantic`` or ``langgraph``.

    The framework modules are imported only when asked for. Running the returned agent
    raises ``AgentUnavailable`` with an installation hint if its framework is missing. An
    unknown name raises ``InputError``, which is also a ``ValueError``.
    """
    if name == "basic":
        from parallelise.agents.basic import BasicAgent

        return BasicAgent()
    if name == "pydantic":
        from parallelise.agents.pydantic_agent import PydanticAgent

        return PydanticAgent()
    if name == "langgraph":
        from parallelise.agents.langgraph_agent import LangGraphAgent

        return LangGraphAgent()
    raise InputError(f"The agent {name!r} is not one of {', '.join(AGENT_NAMES)}. Choose one.")
