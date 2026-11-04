"""Tests that the framework versions are pinned in pyproject.toml, and to what is installed."""

import tomllib
from importlib import metadata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def test_every_framework_requirement_is_pinned_exactly():
    for requirement in PROJECT["optional-dependencies"]["agents"]:
        assert "==" in requirement, f"{requirement} is not pinned to one version"


def test_pins_match_the_versions_installed_here():
    for requirement in PROJECT["optional-dependencies"]["agents"]:
        name, _, pinned = requirement.partition("==")
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            pytest.skip(f"{name} is not installed, so its pin cannot be compared")
        assert installed == pinned, f"{name} is pinned to {pinned} but {installed} is installed"


def test_the_core_dependency_keeps_a_lower_bound_for_httpx():
    assert any(requirement.startswith("httpx>=") for requirement in PROJECT["dependencies"])


def test_the_image_installs_from_the_same_project_file():
    dockerfile = (ROOT / "Dockerfile").read_text()
    installs = [ln for ln in dockerfile.splitlines() if ln.lstrip().startswith("RUN pip install")]
    assert installs and all('".[' in ln and "agents" in ln for ln in installs)
    assert "COPY . /app" in dockerfile  # so pyproject.toml, with its pins, is in the image
