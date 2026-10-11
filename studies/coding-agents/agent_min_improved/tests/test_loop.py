"""Tests for the finish gate and repeat-nudge (fake model, no network)."""
from agent_min_improved import loop


def make_call(name, args_json, cid="1"):
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": args_json}}


def scripted(responses):
    """A fake chat() that returns each scripted response in turn."""
    it = iter(responses)

    def fake(messages, tools=None, **kw):
        return next(it)

    return fake


def test_finish_block_reason_progression():
    assert "have not changed" in loop.finish_block_reason(False, False)
    assert "have not rerun" in loop.finish_block_reason(True, False)
    assert loop.finish_block_reason(True, True) is None


def test_finish_gate_forces_edit_then_verify(tmp_path):
    (tmp_path / "f.txt").write_text("a")
    # 1: tries to finish with no edit -> blocked, nudged
    # 2: edits the file
    # 3: runs tests (verifies)
    # 4: finishes -> now allowed
    responses = [
        {"role": "assistant", "content": "all done"},
        {"role": "assistant", "content": None,
         "tool_calls": [make_call("edit_file", '{"path": "f.txt", "old": "a", "new": "b"}')]},
        {"role": "assistant", "content": None,
         "tool_calls": [make_call("run_tests", "{}")]},
        {"role": "assistant", "content": "fixed it"},
    ]
    out = loop.run_agent("fix it", str(tmp_path),
                         test_cmd=["python", "-c", "print('ok')"],
                         chat=scripted(responses))
    assert out == "fixed it"
    assert (tmp_path / "f.txt").read_text() == "b"      # the gate forced the edit


def test_finish_gate_yields_after_max_nudges(tmp_path):
    # model never edits; the gate nudges max_finish_nudges times, then yields
    responses = [{"role": "assistant", "content": "I think the fix is X"}] * 6
    out = loop.run_agent("fix it", str(tmp_path), chat=scripted(responses),
                         max_finish_nudges=2)
    assert out == "I think the fix is X"


def test_repeat_nudge_then_stops(tmp_path):
    same = {"role": "assistant", "content": None,
            "tool_calls": [make_call("ls", "{}")]}
    out = loop.run_agent("x", str(tmp_path), chat=scripted([same] * 8),
                         max_repeat_nudges=2)
    assert "repeated the same tool call despite nudges" in out
