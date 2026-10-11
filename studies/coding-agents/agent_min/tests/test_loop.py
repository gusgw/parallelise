"""Tests for agent_min.loop with a fake model (no network)."""
from agent_min import loop
from agent_min.tools import build_handlers


def make_call(name, args_json, cid="1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args_json}}


def test_dispatch_unknown_tool():
    out = loop.dispatch_tool_call(make_call("nope", "{}"), {})
    assert out.startswith("ERROR: unknown tool")


def test_dispatch_bad_json(tmp_path):
    handlers = build_handlers(str(tmp_path))
    out = loop.dispatch_tool_call(make_call("ls", "{bad"), handlers)
    assert "bad arguments JSON" in out


def test_dispatch_runs_handler(tmp_path):
    handlers = build_handlers(str(tmp_path))
    out = loop.dispatch_tool_call(
        make_call("write_file", '{"path": "a.txt", "content": "hi"}'), handlers
    )
    assert "wrote" in out


def test_run_agent_uses_tool_then_finishes(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    steps = []

    def fake_chat(messages, tools=None, **kw):
        steps.append(messages)
        if len(steps) == 1:
            return {"role": "assistant", "content": None,
                    "tool_calls": [make_call("read_file", '{"path": "a.txt"}')]}
        return {"role": "assistant", "content": "the file says hello"}

    out = loop.run_agent("read a.txt", str(tmp_path), chat=fake_chat)
    assert out == "the file says hello"
    assert any(m.get("role") == "tool" and "hello" in m.get("content", "")
               for m in steps[-1])


def test_run_agent_hits_step_limit(tmp_path):
    counter = {"n": 0}

    def varying_tool(messages, tools=None, **kw):
        counter["n"] += 1  # distinct args each step, so the repetition guard won't trip
        return {"role": "assistant", "content": None,
                "tool_calls": [make_call("ls", '{"path": "d%d"}' % counter["n"])]}

    out = loop.run_agent("loop", str(tmp_path), chat=varying_tool, max_steps=3)
    assert "max_steps=3" in out


def test_run_agent_repetition_guard(tmp_path):
    def same_tool(messages, tools=None, **kw):
        return {"role": "assistant", "content": None,
                "tool_calls": [make_call("ls", "{}")]}

    out = loop.run_agent("loop", str(tmp_path), chat=same_tool, max_steps=10)
    assert "repeated" in out
