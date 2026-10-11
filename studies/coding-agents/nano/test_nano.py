"""Tiny tests for nano_agent — fast and offline (the loop runs against a fake
model, so no endpoint is needed). Their job is to catch small mistakes:
every function is exercised, and the tool names are checked against dispatch.

    cd nano && python -m pytest -q      # or: bash check.sh
"""
import json

import nano_agent as na


def test_read_write_roundtrip(tmp_path):
    msg = na.write_file(str(tmp_path), "a.txt", "hello")
    assert "wrote" in msg
    assert na.read_file(str(tmp_path), "a.txt") == "hello"


def test_run_tests_reports_exit(tmp_path):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert 1 == 1\n")
    assert "exit 0" in na.run_tests(str(tmp_path))


def test_tool_names_match_dispatch(tmp_path):
    # the names the model sees must be exactly the ones dispatch handles
    assert [t["function"]["name"] for t in na.TOOLS] == \
        ["read_file", "write_file", "run_tests"]
    assert "wrote" in na.dispatch(str(tmp_path), "write_file",
                                  {"path": "b.txt", "content": "x"})
    assert "unknown tool" in na.dispatch(str(tmp_path), "nope", {})


def test_loop_runs_tool_then_finishes(tmp_path, monkeypatch):
    # a fake model: first turn writes a file via a tool call, second turn ends.
    script = iter([
        {"tool_calls": [{"id": "1", "function": {
            "name": "write_file",
            "arguments": json.dumps({"path": "x.txt", "content": "hi"})}}]},
        {"content": "done"},
    ])
    monkeypatch.setattr(na, "chat", lambda messages, tools: next(script))
    result = na.run_agent("write x", str(tmp_path), max_steps=5)
    assert result == "done"
    assert (tmp_path / "x.txt").read_text() == "hi"


def test_loop_stops_at_step_budget(tmp_path, monkeypatch):
    # a model that always calls a tool must still stop at max_steps, not loop forever
    monkeypatch.setattr(na, "chat", lambda messages, tools: {
        "tool_calls": [{"id": "1", "function": {"name": "run_tests", "arguments": "{}"}}]})
    monkeypatch.setattr(na, "run_tests", lambda root: "exit 1")
    assert "max_steps" in na.run_agent("loop", str(tmp_path), max_steps=3)
