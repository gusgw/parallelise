"""Offline tests for the LangGraph starters: the model builder and both flavours
construct without error (no endpoint is contacted — agents are built, not invoked).
Construction is what catches the wiring/import typos.

    cd nano_langgraph && python -m pytest -q
"""
import lg_agent as lg


def test_build_model_constructs(monkeypatch):
    monkeypatch.setenv("MODEL_BASE_URL", "http://x/v1")
    monkeypatch.setenv("MODEL_NAME", "m")
    assert lg.build_model() is not None


def test_build_plain_constructs(tmp_path):
    assert lg.build_plain(str(tmp_path)) is not None


def test_build_deep_constructs(tmp_path):
    assert lg.build_deep(str(tmp_path)) is not None
