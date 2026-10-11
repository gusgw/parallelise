"""A minimal synchronous agent loop: chat -> dispatch tool calls -> chat."""
from __future__ import annotations

import json
from typing import Any, Callable

from harness_common import model, prompts

from agent_min.tools import build_handlers, tool_schemas

# NOTE: keep this SHORT. On Mistral-7B a long/forceful "you MUST call tools"
# prompt backfires — the model narrates compliance ("I used the tool") instead of
# emitting a tool call. See common/MODEL_NOTES.md.
DEFAULT_SYSTEM = (
    "You can read and edit repository files and run tests using the provided "
    "tools. Call a tool to get real data; never invent file contents or claim to "
    "have done something you did not. " + prompts.STRATEGY +
    " When finished, reply with a short summary and no tool call."
)


def dispatch_tool_call(call: dict[str, Any], handlers: dict[str, Callable]) -> str:
    """Run one tool call; return its result as text (errors are returned, not raised)."""
    name = call["function"]["name"]
    handler = handlers.get(name)
    if handler is None:
        return "ERROR: unknown tool " + name
    try:
        args = json.loads(call["function"].get("arguments") or "{}")
    except json.JSONDecodeError as exc:
        return "ERROR: bad arguments JSON: " + str(exc)
    try:
        result = handler(**args)
    except Exception as exc:  # tool errors must feed back to the model, not crash the loop
        return "ERROR: " + str(exc)
    return result if isinstance(result, str) else json.dumps(result)


def run_agent(
    task: str,
    root: str,
    *,
    test_cmd: list[str] | None = None,
    system: str = DEFAULT_SYSTEM,
    max_steps: int = 20,
    usage: dict | None = None,
    chat: Callable = model.chat,
) -> str:
    """Drive the chat/tool loop until a final text answer or the step limit."""
    schemas = tool_schemas()
    handlers = build_handlers(root, test_cmd)
    # Seed with the repo's top-level listing so the model doesn't guess paths.
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompts.with_repo_listing(task, root)},
    ]
    prev_sig = None
    for _ in range(max_steps):
        msg = chat(messages, tools=schemas, usage=usage)  # temperature from MODEL_TEMPERATURE
        calls = msg.get("tool_calls")
        if not calls:
            return msg.get("content") or ""
        # Progress guard: at temperature 0 a failing call repeats forever.
        sig = [(c["function"]["name"], c["function"].get("arguments")) for c in calls]
        if sig == prev_sig:
            return "(stopped: repeated the same tool call without progress)"
        prev_sig = sig
        fixed = model.relabel_tool_calls(calls)
        messages.append(
            {"role": "assistant", "content": msg.get("content") or "", "tool_calls": fixed}
        )
        for call in fixed:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": dispatch_tool_call(call, handlers),
                }
            )
    return "(stopped: reached max_steps=" + str(max_steps) + ")"
