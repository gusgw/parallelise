"""Registry of the target codebases the agent can work on.

The Target type and lookup live here; the actual entries are loaded from
harness_common.targets_local if that module is present. Without it TARGETS is
empty, which is how the unit tests run.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Target:
    name: str
    path: str
    apps: list[str] = field(default_factory=list)
    test_cmd: list[str] = field(default_factory=lambda: ["python", "-m", "pytest", "-q"])


def _load_targets() -> dict[str, "Target"]:
    """Load the study's targets if the module that lists them is present, else none."""
    try:
        from harness_common import targets_local
    except ImportError:
        return {}
    return dict(targets_local.TARGETS)


TARGETS = _load_targets()


def get_target(name: str) -> Target:
    """Return the Target for name; HARNESS_TARGET_PATH overrides its path (worktrees)."""
    if name not in TARGETS:
        raise ValueError("unknown target: " + name + " (have: " + ", ".join(TARGETS) + ")")
    t = TARGETS[name]
    override = os.environ.get("HARNESS_TARGET_PATH")
    if override:
        return Target(name=t.name, path=override, apps=list(t.apps), test_cmd=list(t.test_cmd))
    return t
