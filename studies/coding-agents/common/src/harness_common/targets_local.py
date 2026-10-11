"""The target codebases used in the study.

Loaded by harness_common.targets._load_targets() if present. Each target is a small
Python project that the agents work on. The projects are not part of this
repository: place them in a directory called "targets" beside where you run the
harness, or set HARNESS_TARGET_ROOT to the directory that holds them.
"""
from __future__ import annotations

import os

from harness_common.targets import Target

TARGET_ROOT = os.environ.get("HARNESS_TARGET_ROOT", "targets")

TARGETS = {
    "mealworm": Target(
        name="mealworm",
        path=TARGET_ROOT + "/mealworm",
        apps=["mealworm.pipeline_inline:app", "mealworm.pipeline_external:app"],
        test_cmd=[".venv/bin/python", "-m", "pytest", "-q"],
    ),
    "best_weather": Target(
        name="best_weather",
        path=TARGET_ROOT + "/best_weather",
        apps=["pipeline:app"],
        test_cmd=[".venv/bin/python", "-m", "pytest", "-q"],
    ),
    "weather": Target(
        name="weather",
        path=TARGET_ROOT + "/weather",
        apps=["pipeline:app"],
        test_cmd=[".venv/bin/python", "-m", "pytest", "-q"],
    ),
}
