"""The plain agent: one loop, one HTTP client, no framework.

This is the baseline design. It calls the model through the shared ``ModelClient``, runs the
tools the model asks for, and, whenever the model says it has finished, runs the toolset's
check itself. Only a passing check ends the run. Otherwise the failure is fed back to the
model, so that a fix which was described in words but never written to disk cannot end it.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from parallelise.agents.base import AgentResult, Toolset
from parallelise.errors import ModelError

if TYPE_CHECKING:
    from parallelise.config import Settings
    from parallelise.model import ModelClient

log = logging.getLogger("parallelise.agents.basic")


class BasicAgent:
    """The plain loop over the shared model client."""

    def run(self, work_dir: Path, task: str, settings: Settings, client: ModelClient,
            step_budget: int, toolset: Toolset, system_prompt: str) -> AgentResult:
        """Loop until the work passes the toolset's check, ``step_budget`` model calls have been
        made, or the time limit in ``settings`` has passed, which is checked before every call.
        """
        deadline = time.monotonic() + settings.time_limit_s
        messages: list[dict] = [{"role": "system", "content": system_prompt},
                                {"role": "user", "content": task}]  # fmt: skip
        for step in range(1, step_budget + 1):
            if time.monotonic() >= deadline:
                log.warning("step %d: the time limit has passed; stopping the agent", step)
                return AgentResult(False, time_limit_reached=True)
            try:
                message = client.chat(messages, toolset.schemas)
            except ModelError as exc:
                log.warning("step %d: the model call failed: %s", step, exc)
                return AgentResult(False, error=f"{type(exc).__name__}: {exc}")
            messages.append(message)
            calls = message.get("tool_calls")
            if calls:
                messages.extend(self._run_call(toolset, call) for call in calls)
                continue
            passed, feedback = toolset.finish()
            if passed:
                return AgentResult(True, "the checks passed")
            log.info("step %d: the model finished but the checks fail; asking again", step)
            messages.append({"role": "user", "content": feedback})
        return AgentResult(False, "stopped: the step budget was spent without passing checks")

    @staticmethod
    def _run_call(toolset: Toolset, call: dict) -> dict:
        """Run one tool call from the model and build the tool message that answers it."""
        function = call.get("function", {})
        try:
            output = toolset.run(function.get("name", ""),
                                 json.loads(function.get("arguments") or "{}"))  # fmt: skip
        except (ValueError, AttributeError, TypeError) as exc:
            output = f"error: could not read the tool arguments: {exc}"
        return {"role": "tool", "tool_call_id": call.get("id", ""), "content": str(output)}
