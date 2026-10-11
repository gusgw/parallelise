"""Tests for agent_langgraph.agent (construction only — no live model)."""
from langchain_openai import ChatOpenAI

from agent_langgraph import agent


def test_build_model_uses_config(monkeypatch):
    monkeypatch.setenv("MODEL_BASE_URL", "http://x/v1")
    monkeypatch.setenv("MODEL_NAME", "foo")
    monkeypatch.setenv("MODEL_API_KEY", "")
    m = agent.build_model()
    assert isinstance(m, ChatOpenAI)
    assert (getattr(m, "model_name", None) or getattr(m, "model", None)) == "foo"


def test_build_agent_returns_invokable(tmp_path):
    a = agent.build_agent(str(tmp_path))
    assert hasattr(a, "invoke")


def test_build_agent_profile_trims_meta_tools(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_BASE_URL", "http://x/v1")
    monkeypatch.setenv("MODEL_NAME", "foo")
    monkeypatch.setenv("MODEL_API_KEY", "")
    a = agent.build_agent(str(tmp_path))
    tools = a.nodes["tools"].bound.tools_by_name
    names = set(tools.keys())
    assert "write_todos" not in names and "task" not in names   # trimmed
    assert {"read_file", "edit_file", "execute"} <= names       # essentials kept
    # read_file/edit_file are OUR raw-text overrides, not the middleware's
    assert "raw text" in tools["read_file"].description
    assert "quotes the real file text" in tools["edit_file"].description


def test_override_tools_normalize_slash_paths(tmp_path):
    (tmp_path / "f.txt").write_text("hello\n")
    read_file, edit_file = agent._override_tools(str(tmp_path))
    assert read_file("/f.txt") == "hello\n"                 # leading slash ok
    assert "replaced" in edit_file("/f.txt", "hello", "bye")
    assert (tmp_path / "f.txt").read_text() == "bye\n"
    assert read_file("/missing.txt").startswith("ERROR:")   # errors as strings


def test_override_edit_accepts_content_alias(tmp_path):
    (tmp_path / "f.txt").write_text("aaa\n")
    _, edit_file = agent._override_tools(str(tmp_path))
    assert "replaced" in edit_file("/f.txt", "aaa", content="bbb")   # write_file-style arg
    assert (tmp_path / "f.txt").read_text() == "bbb\n"
    assert edit_file("/f.txt", "bbb").startswith("ERROR: pass new_string")


def test_build_agent_prompt_names_test_command(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_NAME", "foo")
    # the test-command hint is composed into the system prompt
    hint = agent._test_hint(["x/bin/py", "-m", "pytest"])
    assert 'execute(command="x/bin/py -m pytest")' in hint


def test_build_smalltools_agent_invokable(tmp_path):
    a = agent.build_smalltools_agent(str(tmp_path))
    assert hasattr(a, "invoke")
    names = set(a.nodes["tools"].bound.tools_by_name.keys())
    assert {"read_file", "ls", "grep", "edit_file", "write_file", "run_tests"} <= names
