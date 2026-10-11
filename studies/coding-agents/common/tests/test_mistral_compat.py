"""Tests for harness_common.mistral_compat (pure id-relabel logic)."""
from harness_common import mistral_compat


def test_relabel_consistent_and_9char():
    body = {"messages": [
        {"role": "assistant", "tool_calls": [
            {"id": "1", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "x"},
    ]}
    out = mistral_compat.relabel_ids_in_body(body)
    aid = out["messages"][0]["tool_calls"][0]["id"]
    tid = out["messages"][1]["tool_call_id"]
    assert aid == tid                       # assistant id and tool ref stay matched
    assert len(aid) == 9 and aid.isalnum()
    assert aid != "1"


def test_relabel_distinct_ids():
    body = {"messages": [
        {"role": "assistant", "tool_calls": [
            {"id": "1", "function": {"name": "a", "arguments": "{}"}},
            {"id": "2", "function": {"name": "b", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "content": "x"},
        {"role": "tool", "tool_call_id": "2", "content": "y"},
    ]}
    out = mistral_compat.relabel_ids_in_body(body)
    a = out["messages"][0]["tool_calls"][0]["id"]
    b = out["messages"][0]["tool_calls"][1]["id"]
    assert a != b
    assert out["messages"][1]["tool_call_id"] == a
    assert out["messages"][2]["tool_call_id"] == b


def test_relabel_ignores_plain_messages():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert mistral_compat.relabel_ids_in_body(body) == {"messages": [{"role": "user", "content": "hi"}]}
