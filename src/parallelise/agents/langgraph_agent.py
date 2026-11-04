"""The LangGraph agent: a tool-calling graph inside a re-check loop.

LangChain's ``create_agent`` builds a graph that alternates between the model and the tools
until the model stops asking for tools. That alone would let the model finish by describing
a fix, so each full pass of the graph is followed by a run of the checks; if they fail, the
conversation continues with the failure appended, up to a fixed number of passes.
The frameworks are imported lazily so that the package works without them.

The graph is streamed state by state, which lets the agent stop between steps when the time
limit has passed, and keeps the messages (and with them the counts of requests and tokens)
when a run ends in an error. The model is reached through HTTP clients that wait out HTTP 429
answers like the other agents; the OpenAI library's own retries are off so that they do not
repeat that waiting.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from parallelise.agents import tools
from parallelise.agents.base import (
    AgentResult,
    AgentUnavailable,
    Deadline,
    LatencyRecorder,
    build_system_prompt,
    check_feedback,
)
from parallelise.model import ThrottledAsyncClient, ThrottledClient

if TYPE_CHECKING:
    from parallelise.config import Settings
    from parallelise.model import ModelClient

log = logging.getLogger("parallelise.agents.langgraph")

INSTALL_HINT = "install the frameworks with: pip install 'parallelise[agents]'"
MAX_PASSES = 8


def build_model(settings: Settings, recorder: LatencyRecorder | None = None):
    """Build the LangChain chat model that talks to ``settings.gateway_url``."""
    try:
        from langchain_openai import ChatOpenAI
        from pydantic import SecretStr
    except ImportError as exc:
        raise AgentUnavailable(
            f"the langgraph agent needs langchain-openai; {INSTALL_HINT}"
        ) from exc
    sync_client = ThrottledClient(
        throttle_wait_s=settings.throttle_wait_s,
        timeout=settings.request_timeout_s,
        event_hooks=recorder.sync_hooks() if recorder else None,
    )
    async_client = ThrottledAsyncClient(
        throttle_wait_s=settings.throttle_wait_s,
        timeout=settings.request_timeout_s,
        event_hooks=recorder.async_hooks() if recorder else None,
    )
    return ChatOpenAI(
        base_url=settings.gateway_url,
        api_key=SecretStr(settings.api_key or "not-needed"),
        model=settings.model,
        temperature=settings.temperature,
        http_client=sync_client,
        http_async_client=async_client,
        max_retries=0,
    )


def _make_tools(work_dir: Path) -> list:
    """The four shared tools as plain functions; LangChain reads their signatures."""

    def read_notebook(path: str) -> str:
        """Read a Jupyter notebook as plain text (cells in order)."""
        return tools.read_notebook(work_dir, path)

    def read_file(path: str) -> str:
        """Read a text file in the work folder."""
        return tools.read_file(work_dir, path)

    def write_file(path: str, content: str) -> str:
        """Create or overwrite a file in the work folder."""
        return tools.write_file(work_dir, path, content)

    def run_shell(command: str) -> str:
        """Run an allow-listed build or test command (make, gcc, ./binary, ...)."""
        return tools.run_shell(work_dir, command)

    return [read_notebook, read_file, write_file, run_shell]


def build_agent(work_dir: Path, settings: Settings, recorder: LatencyRecorder | None = None):
    """Build the tool-calling graph for one work folder."""
    try:
        from langchain.agents import create_agent
    except ImportError as exc:
        raise AgentUnavailable(f"the langgraph agent needs langchain; {INSTALL_HINT}") from exc
    work_dir = Path(work_dir)
    return create_agent(
        model=build_model(settings, recorder),
        system_prompt=build_system_prompt(work_dir),
        tools=_make_tools(work_dir),
    )


def _count_usage(messages: list) -> tuple[int, int, int]:
    """Sum (requests, prompt tokens, completion tokens) over the model's replies."""
    requests = prompt = completion = 0
    for message in messages:
        if getattr(message, "type", "") != "ai":
            continue
        requests += 1
        usage = getattr(message, "usage_metadata", None) or {}
        prompt += int(usage.get("input_tokens", 0))
        completion += int(usage.get("output_tokens", 0))
    return requests, prompt, completion


def _model_is_called_next(state: dict) -> bool:
    """Say whether the graph, after reaching ``state``, goes on to call the model.

    It does unless the last message is the model's answer with no tool calls, which ends the
    pass. The time limit is only enforced when it would stop a model call, so a pass that is
    already ending is not reported as stopped by it.
    """
    messages = state.get("messages") or []
    if not messages:
        return True
    last = messages[-1]
    if getattr(last, "type", "") != "ai":
        return True
    return bool(getattr(last, "tool_calls", None))


class LangGraphAgent:
    """LangChain's tool-calling graph with a loop that re-runs the checks."""

    name = "langgraph"

    def run(
        self,
        work_dir: Path,
        task: str,
        settings: Settings,
        client: ModelClient,
        step_budget: int = 80,
    ) -> AgentResult:
        """Run up to ``MAX_PASSES`` passes of the graph, each limited to ``step_budget`` steps.

        The graph is streamed state by state. Between states the time limit in ``settings`` is
        checked, so a run past its limit stops before its next model call. A failure of the
        model service or the framework ends the run, and the result holds the error with the
        counts of the requests that were made before it.
        """
        try:
            from langgraph.errors import GraphRecursionError
        except ImportError as exc:
            raise AgentUnavailable(f"the langgraph agent needs langgraph; {INSTALL_HINT}") from exc
        work_dir = Path(work_dir)
        deadline = Deadline(settings.time_limit_s)
        recorder = LatencyRecorder()
        agent = build_agent(work_dir, settings, recorder)

        messages: list = [{"role": "user", "content": task}]
        notes = ""
        error = ""
        finished = False
        time_limit_reached = False
        passes = 0
        for passes in range(1, MAX_PASSES + 1):
            state: dict = {"messages": messages}
            try:
                stream = agent.stream(
                    {"messages": messages},
                    config={"recursion_limit": step_budget},
                    stream_mode="values",
                )
                # Streaming the full state keeps the messages even if the step limit is hit.
                with contextlib.closing(stream):
                    for state in stream:  # noqa: B007 - the last state is used below
                        if deadline.expired() and _model_is_called_next(state):
                            time_limit_reached = True
                            break
            except GraphRecursionError:
                messages = list(state["messages"])
                notes = "stopped: a pass reached the step limit"
                break
            except Exception as exc:  # a failed model service must not hide the counts
                messages = list(state["messages"])
                error = f"{type(exc).__name__}: {exc}"
                log.warning("langgraph agent stopped: %s", error)
                break
            messages = list(state["messages"])
            if time_limit_reached:
                log.warning("pass %d: the time limit has passed; stopping the agent", passes)
                break
            passed, output = tools.run_check(work_dir)
            if passed:
                finished = True
                notes = "the checks passed"
                break
            log.info("pass %d: the checks fail; asking again", passes)
            messages.append(
                {
                    "role": "user",
                    "content": check_feedback(list(tools.DEFAULT_CHECK_COMMAND), output),
                }
            )
        else:
            notes = "stopped: the checks never passed"

        requests, prompt, completion = _count_usage(messages)
        return AgentResult(
            finished=finished,
            steps=passes,
            requests=requests,
            prompt_tokens=prompt,
            completion_tokens=completion,
            latencies_s=list(recorder.latencies_s),
            notes=notes,
            time_limit_reached=time_limit_reached,
            error=error,
        )
