"""Guard middlewares for LangGraph agents.

Conversation-log findings these encode:
- neither langchain nor deepagents stops a model that makes the same failing
  tool call over and over (one run repeated an identical grep 17 times to
  recursion death) -> RepeatGuardMiddleware;
- nothing stops a model from ENDING with a prose essay describing the fix
  instead of applying it (one run finished with a 46k-char essay and no edit)
  -> FinishGateMiddleware, the langchain-side equivalent of a Pydantic AI
  output validator.
"""
from __future__ import annotations

import json

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

REPEAT_MSG = (
    "You already made this exact tool call and saw its result; repeating it "
    "returns the same thing. Take a DIFFERENT action: change the arguments or "
    "use another tool."
)


def call_signature(name: str | None, args) -> str:
    """A stable string identity for one tool call."""
    return str(name) + ":" + json.dumps(args, sort_keys=True, default=str)


def previous_call_signature(messages, current_id: str | None) -> str | None:
    """Signature of the most recent already-issued tool call other than current_id."""
    calls = []
    for m in messages or []:
        for c in getattr(m, "tool_calls", None) or []:
            calls.append(c)
    prior = [c for c in calls if c.get("id") != current_id]
    if not prior:
        return None
    last = prior[-1]
    return call_signature(last.get("name"), last.get("args"))


class RepeatGuardMiddleware(AgentMiddleware):
    """Replace an exact-repeat tool call with corrective feedback to the model."""

    def wrap_tool_call(self, request, handler):
        state = request.state
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        call = request.tool_call
        if previous_call_signature(messages, call.get("id")) == call_signature(
                call.get("name"), call.get("args")):
            return ToolMessage(content=REPEAT_MSG, tool_call_id=call["id"],
                               name=call.get("name") or "tool")
        return handler(request)


EDIT_TOOLS = ("edit_file", "write_file")


def unverified_reason(last_edit: int | None, last_test: int | None) -> str | None:
    """Why the agent may not finish yet, or None when it may."""
    if last_edit is None:
        return ("You have not changed anything yet. Apply the fix with "
                "edit_file — do not describe it in text.")
    if last_test is None or last_test < last_edit:
        return ("You edited the code but have not rerun the tests since that "
                "edit. Run the tests now; only finish when they pass.")
    return None


def final_ai_message(response):
    """The last AIMessage of a ModelResponse (or a bare AIMessage), else None."""
    if isinstance(response, AIMessage):
        return response
    result = getattr(response, "result", None) or []
    ais = [m for m in result if isinstance(m, AIMessage)]
    return ais[-1] if ais else None


class FinishGateMiddleware(AgentMiddleware):
    """Refuse a prose finish until an edit was made and tests were rerun after it.

    The langchain loop treats any model response without tool calls as the end
    of the run; this gate re-asks the model (a bounded number of times) with a
    corrective message instead — the hand-rolled equivalent of Pydantic AI's
    output-validator + tool_choice forcing.
    """

    def __init__(self, max_nudges: int = 2):
        super().__init__()
        self._seq = 0
        self._last_edit: int | None = None
        self._last_test: int | None = None
        self._max_nudges = max_nudges

    def wrap_tool_call(self, request, handler):
        result = handler(request)
        name = request.tool_call.get("name")
        args = request.tool_call.get("args") or {}
        self._seq += 1
        if name in EDIT_TOOLS and not str(getattr(result, "content", "")).startswith("Error"):
            self._last_edit = self._seq
        if name == "run_tests" or (name == "execute" and "pytest" in str(args.get("command", ""))):
            self._last_test = self._seq
        return result

    def wrap_model_call(self, request, handler):
        response = handler(request)
        for _ in range(self._max_nudges):
            message = final_ai_message(response)
            if message is None or getattr(message, "tool_calls", None):
                return response
            reason = unverified_reason(self._last_edit, self._last_test)
            if reason is None:
                return response
            # keep the abandoned text in the retried request, then correct it
            request.messages.append(message)
            request.messages.append(HumanMessage(content=reason))
            response = handler(request)
        return response
