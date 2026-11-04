"""Tests for the facts about the software and the machine that go into every run record."""

import re
import shutil
from importlib import metadata
from pathlib import Path

import parallelise
from parallelise import provenance

PACKAGE = Path(parallelise.__file__).resolve().parent


def test_package_version_is_the_packages_own():
    assert provenance.package_version() == parallelise.__version__


def test_framework_versions_list_only_what_is_installed():
    found = provenance.framework_versions()
    assert found["httpx"] == metadata.version("httpx")
    for name, version in found.items():
        assert version == metadata.version(name)
    assert set(found) <= set(provenance.FRAMEWORK_PACKAGES)


def test_framework_versions_skip_a_package_that_is_not_installed(monkeypatch):
    real = metadata.version

    def version(name):
        if name == "langgraph":
            raise metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(provenance.metadata, "version", version)
    assert "langgraph" not in provenance.framework_versions()
    assert "httpx" in provenance.framework_versions()


def test_the_framework_list_names_the_libraries_that_carry_the_requests():
    assert {"httpx", "pydantic-ai", "langgraph", "langchain", "langchain-openai",
            "openai"} <= set(provenance.FRAMEWORK_PACKAGES)


def test_cpu_count_is_a_positive_whole_number():
    assert isinstance(provenance.cpu_count(), int) and provenance.cpu_count() >= 1


def test_source_hash_is_short_and_steady():
    first = provenance.source_hash()
    assert re.fullmatch(r"[0-9a-f]{12}", first)
    assert provenance.source_hash() == first


def test_source_hash_changes_when_any_python_file_changes(tmp_path):
    copy = tmp_path / "parallelise"
    shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    before = provenance.source_hash(copy)
    assert before == provenance.source_hash(PACKAGE)  # the same files, the same hash
    target = copy / "study" / "report.py"  # a file well away from the top of the package
    target.write_text(target.read_text() + "\n# one more comment\n")
    assert provenance.source_hash(copy) != before


def test_source_hash_changes_when_a_file_is_added_or_renamed(tmp_path):
    copy = tmp_path / "parallelise"
    shutil.copytree(PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    before = provenance.source_hash(copy)
    (copy / "extra.py").write_text("")
    added = provenance.source_hash(copy)
    assert added != before
    (copy / "extra.py").rename(copy / "other.py")
    assert provenance.source_hash(copy) not in (before, added)
