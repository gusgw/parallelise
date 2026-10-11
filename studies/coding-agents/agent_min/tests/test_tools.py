"""Tests for agent_min.tools."""
from agent_min import tools


def test_schemas_have_expected_names():
    names = {t["function"]["name"] for t in tools.tool_schemas()}
    assert {"read_file", "ls", "grep", "edit_file", "write_file", "run_tests"} <= names


def test_handlers_read_and_edit(tmp_path):
    h = tools.build_handlers(str(tmp_path))
    h["write_file"](path="x.txt", content="hello")
    assert h["read_file"](path="x.txt") == "hello"
    h["edit_file"](path="x.txt", old="hello", new="bye")
    assert h["read_file"](path="x.txt") == "bye"


def test_grep_handler_no_matches(tmp_path):
    h = tools.build_handlers(str(tmp_path))
    assert h["grep"](pattern="zzz") == "(no matches)"
