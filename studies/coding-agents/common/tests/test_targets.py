"""Tests for harness_common.targets (mechanism only; no target data)."""
import pytest

from harness_common import targets
from harness_common.targets import Target


def _registry():
    return {"demo": Target(name="demo", path="/x", apps=["pkg:app"])}


def test_get_target_returns_target(monkeypatch):
    monkeypatch.setattr(targets, "TARGETS", _registry())
    t = targets.get_target("demo")
    assert t.name == "demo" and t.apps  # non-empty


def test_get_target_unknown_raises(monkeypatch):
    monkeypatch.setattr(targets, "TARGETS", _registry())
    with pytest.raises(ValueError):
        targets.get_target("does-not-exist")


def test_get_target_honors_path_override(monkeypatch):
    monkeypatch.setattr(targets, "TARGETS", _registry())
    monkeypatch.setenv("HARNESS_TARGET_PATH", "/tmp/wt")
    t = targets.get_target("demo")
    assert t.path == "/tmp/wt"
    assert t.name == "demo" and t.test_cmd  # name + test_cmd preserved


def test_load_targets_tolerates_absent_local():
    # Without the targets module the registry is empty, never an error.
    assert isinstance(targets._load_targets(), dict)
