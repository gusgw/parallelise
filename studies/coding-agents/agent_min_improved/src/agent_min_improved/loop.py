"""agent_min's loop plus two ideas ported from the framework agents:

- a FINISH GATE: do not accept a no-tool-call "finish" until an edit was made and
  the tests were rerun after that edit (a bounded number of nudges, then yield);
- a repeat NUDGE: an exact-repeat tool call gets corrective feedback and the loop
  CONTINUES (up to a bound), instead of agent_min's hard abort on the first repeat.

Tools, dispatch, and the seeded prompt are reused verbatim from agent_min, so the
only behavioural difference from the control is these two gates.
"""
from __future__ import annotations

from typing import Callable

from harness_common import model, prompts

from agent_min.loop import DEFAULT_SYSTEM, dispatch_tool_call
from agent_min.tools import build_handlers, tool_schemas

REPEAT_MSG = (
    "You already made this exact call and saw its result; it will not change. "
    "Take a DIFFERENT action: change the arguments or use another tool."
)

EDIT_TOOLS = ("edit_file", "write_file")


def finish_block_reason(edited: bool, verified_after_edit: bool) -> str | None:
    """Why the agent may not finish yet, or None when it may."""
    if not edited:
        return ("You have not changed any file yet. Make the fix with edit_file, "
                "then run the tests. Do not stop before the tests pass.")
    if not verified_after_edit:
        return ("You edited the code but have not rerun the tests since that edit. "
                "Call run_tests now; only finish after it reports results.")
    return None


def run_agent(
    task: str,
    root: str,
    *,
    test_cmd: list[str] | None = None,
    system: str = DEFAULT_SYSTEM,
    max_steps: int = 20,
    max_finish_nudges: int = 3,
    max_repeat_nudges: int = 3,
    usage: dict | None = None,
    chat: Callable = model.chat,
) -> str:
    """Drive the loop with a finish gate + repeat-nudge; same tools as agent_min."""
    schemas = tool_schemas()
    handlers = build_handlers(root, test_cmd)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompts.with_repo_listing(task, root)},
    ]
    prev_sig = None
    consecutive_repeats = 0
    edited = False
    verified_after_edit = False
    finish_nudges = 0
    for _ in range(max_steps):
        msg = chat(messages, tools=schemas, usage=usage)  # temperature from MODEL_TEMPERATURE
        calls = msg.get("tool_calls")
        if not calls:
            reason = finish_block_reason(edited, verified_after_edit)
            if reason is None or finish_nudges >= max_finish_nudges:
                return msg.get("content") or ""
            finish_nudges += 1
            messages.append({"role": "assistant", "content": msg.get("content") or ""})
            messages.append({"role": "user", "content": reason})
            continue
        sig = [(c["function"]["name"], c["function"].get("arguments")) for c in calls]
        fixed = model.relabel_tool_calls(calls)
        messages.append(
            {"role": "assistant", "content": msg.get("content") or "", "tool_calls": fixed}
        )
        if sig == prev_sig:
            consecutive_repeats += 1
            for call in fixed:  # keep the message structure valid
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": REPEAT_MSG})
            if consecutive_repeats > max_repeat_nudges:
                return "(stopped: repeated the same tool call despite nudges)"
            continue
        consecutive_repeats = 0
        prev_sig = sig
        for call in fixed:
            result = dispatch_tool_call(call, handlers)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
            name = call["function"]["name"]
            if name in EDIT_TOOLS and not result.startswith("ERROR"):
                edited = True
                verified_after_edit = False
            if name == "run_tests" and edited:
                verified_after_edit = True
    return "(stopped: reached max_steps=" + str(max_steps) + ")"
