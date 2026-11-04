"""The plain agent: one loop, one HTTP client, no framework.

This is the baseline design. It calls the model through the shared ``ModelClient``, runs the
tools the model asks for, and, whenever the model says it has finished, runs the checks
itself. Only a passing check ends the run. Otherwise the failure is fed back to the model,
so that a fix which was described in words but never written to disk cannot end the run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from parallelise.agents import tools
from parallelise.agents.base import AgentResult, Deadline, build_system_prompt, check_feedback
from parallelise.errors import ModelError

if TYPE_CHECKING:
    from parallelise.config import Settings
    from parallelise.model import ModelClient

log = logging.getLogger("parallelise.agents.basic")


class BasicAgent:
    """The plain loop over the shared model client."""

    name = "basic"

    def run(
        self,
        work_dir: Path,
        task: str,
        settings: Settings,
        client: ModelClient,
        step_budget: int = 80,
    ) -> AgentResult:
        """Loop until the checks pass, ``step_budget`` model calls have been made, or the time
        limit in ``settings`` has passed.

        The time is checked before every call to the model. If the model service fails, the
        run ends there and the result carries the error with the counts so far.
        """
        work_dir = Path(work_dir)
        deadline = Deadline(settings.time_limit_s)
        messages: list[dict] = [
            {"role": "system", "content": build_system_prompt(work_dir)},
            {"role": "user", "content": task},
        ]
        result = AgentResult(False, 0, 0, 0, 0, [], "")
        for step in range(1, step_budget + 1):
            if deadline.expired():
                result.time_limit_reached = True
                log.warning("step %d: the time limit has passed; stopping the agent", step)
                return result
            try:
                reply = client.chat(messages, tools.TOOL_SCHEMAS)
            except ModelError as exc:
                result.error = f"{type(exc).__name__}: {exc}"
                log.warning("step %d: the model call failed: %s", step, exc)
                return result
            result.steps = step
            result.requests += 1
            result.prompt_tokens += reply.prompt_tokens
            result.completion_tokens += reply.completion_tokens
            result.latencies_s.append(reply.latency_s)

            message = dict(reply.message)
            if message.get("content") is None:
                message["content"] = ""
            messages.append(message)

            calls = message.get("tool_calls")
            if calls:
                for call in calls:
                    messages.append(self._run_call(work_dir, call))
                continue

            passed, output = tools.run_check(work_dir)
            if passed:
                result.finished = True
                result.notes = "the checks passed"
                return result
            log.info("step %d: the model finished but the checks fail; asking again", step)
            messages.append(
                {
                    "role": "user",
                    "content": check_feedback(list(tools.DEFAULT_CHECK_COMMAND), output),
                }
            )
        result.notes = "stopped: the step budget was spent without passing checks"
        return result

    @staticmethod
    def _run_call(work_dir: Path, call: dict) -> dict:
        """Run one tool call from the model and build the tool message that answers it."""
        function = call.get("function", {})
        try:
            args = json.loads(function.get("arguments") or "{}")
            output = tools.dispatch(work_dir, function.get("name", ""), args)
        except (ValueError, AttributeError) as exc:
            output = f"error: could not read the tool arguments: {exc}"
        return {"role": "tool", "tool_call_id": call.get("id", ""), "content": str(output)}
