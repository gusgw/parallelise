"""Tests for the repeat-call guard (pure helpers — no graph, no model)."""
from dataclasses import dataclass, field

from agent_langgraph import middleware


@dataclass
class FakeAI:
    tool_calls: list = field(default_factory=list)


def test_call_signature_stable_across_key_order():
    a = middleware.call_signature("grep", {"pattern": "x", "path": "."})
    b = middleware.call_signature("grep", {"path": ".", "pattern": "x"})
    assert a == b


def test_previous_call_signature_skips_current():
    msgs = [FakeAI([{"name": "grep", "args": {"p": 1}, "id": "a"}]),
            FakeAI([{"name": "grep", "args": {"p": 1}, "id": "b"}])]
    # the previous call (id=a) has the same signature as the current (id=b)
    prev = middleware.previous_call_signature(msgs, current_id="b")
    assert prev == middleware.call_signature("grep", {"p": 1})


def test_previous_call_signature_none_when_first():
    msgs = [FakeAI([{"name": "ls", "args": {}, "id": "a"}])]
    assert middleware.previous_call_signature(msgs, current_id="a") is None


def test_different_args_not_flagged():
    msgs = [FakeAI([{"name": "grep", "args": {"p": 1}, "id": "a"}]),
            FakeAI([{"name": "grep", "args": {"p": 2}, "id": "b"}])]
    prev = middleware.previous_call_signature(msgs, current_id="b")
    assert prev != middleware.call_signature("grep", {"p": 2})


def test_unverified_reason_progression():
    assert "not changed" in middleware.unverified_reason(None, None)
    assert "rerun the tests" in middleware.unverified_reason(3, None)   # edit, no test
    assert "rerun the tests" in middleware.unverified_reason(5, 2)      # test before edit
    assert middleware.unverified_reason(2, 5) is None                   # verified


def test_final_ai_message_shapes():
    from langchain_core.messages import AIMessage, ToolMessage

    bare = AIMessage(content="hi")
    assert middleware.final_ai_message(bare) is bare

    class FakeResponse:
        result = [ToolMessage(content="t", tool_call_id="x"), AIMessage(content="end")]

    assert middleware.final_ai_message(FakeResponse()).content == "end"

    class Empty:
        result = []

    assert middleware.final_ai_message(Empty()) is None
