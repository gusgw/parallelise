"""Probe the live model for tool-calling support (Phase 1.2 go/no-go gate).

Run from a host that can reach MODEL_BASE_URL:
    cd common && PYTHONPATH=src python probe_model.py
"""
from __future__ import annotations

from harness_common import model

ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two integers and return the sum.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    },
}


def main() -> None:
    cfg = model.model_config()
    print("endpoint:", cfg["base_url"], "| model:", cfg["model"])

    # 1) plain completion — confirms basic chat works
    msg = model.chat(
        [{"role": "user", "content": "Reply with the single word: ready"}],
        params={"temperature": 0},
    )
    print("\n[1] plain reply:", repr(msg.get("content")))

    # 2) tool-calling — does the model emit a well-formed tool_call?
    messages = [{"role": "user", "content": "What is 21 + 21? Use the add tool."}]
    msg = model.chat(messages, tools=[ADD_TOOL], params={"temperature": 0})
    calls = msg.get("tool_calls")
    print("\n[2] tool_calls present:", bool(calls))
    if not calls:
        print("    content:", repr(msg.get("content")))
        print("\nRESULT: model did NOT emit tool_calls — tool calling off/unreliable.")
        return
    # 3) feed the tool result back (ids relabelled to satisfy the Mistral template)
    fixed = model.relabel_tool_calls(calls)
    call = fixed[0]
    print("    name:", call["function"]["name"], "| args:", call["function"]["arguments"])
    messages.append(
        {"role": "assistant", "content": msg.get("content") or "", "tool_calls": fixed}
    )
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": "42"})
    final = model.chat(messages, tools=[ADD_TOOL], params={"temperature": 0})
    print("\n[3] final reply:", repr(final.get("content")))
    print("\nRESULT: tool-calling round trip OK.")


if __name__ == "__main__":
    main()
