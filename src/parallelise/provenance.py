"""Facts about the software and the machine, noted in every run record.

A result is only as useful as the account of what produced it. These small functions say which
version of parallelise ran (and, by a hash of its source files, whether the code has changed
since), which versions of the libraries that carry the requests were installed, and how many
processor cores the run could use, which bears on the speed-ups the benchmark measures.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from importlib import metadata
from pathlib import Path

from . import __version__

#: The libraries whose versions are noted when they are installed. The first six are the
#: ones named in the project's requirements; httpx2 is the HTTP library that Pydantic AI and
#: the OpenAI library now use underneath.
FRAMEWORK_PACKAGES = (
    "httpx",
    "httpx2",
    "pydantic-ai",
    "langgraph",
    "langchain",
    "langchain-openai",
    "openai",
)

#: Length of the source hash, in hexadecimal digits.
HASH_LENGTH = 12

#: What counts as source: the Python files and the C helper files that every work folder gets.
_SOURCE_SUFFIXES = (".py", ".h", ".sh")


def package_version() -> str:
    """Return the version of parallelise itself."""
    return __version__


def framework_versions() -> dict[str, str]:
    """Return the installed version of each package in ``FRAMEWORK_PACKAGES``.

    A package that is not installed is left out.
    """
    found: dict[str, str] = {}
    for name in FRAMEWORK_PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def cpu_count() -> int:
    """Return the number of processor cores this process may use, and at least 1.

    The count of cores the process is allowed to run on is used where the system gives it,
    since that is what limits a benchmark in a container or under a core-pinning command.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def source_hash(package_dir: Path | None = None) -> str:
    """Return a short hash of the source files of the installed package.

    Every ``.py`` file in the package is covered, and so are the C and shell helpers in
    ``data/`` that are copied into every work folder, because they too decide what a run
    does. Each file's path (relative to the package) goes into the hash with its contents, so
    that adding, removing or renaming a file changes the hash as well as editing one.
    """
    if package_dir is None:
        return _installed_source_hash()
    return _hash_sources(Path(package_dir))


@lru_cache(maxsize=1)
def _installed_source_hash() -> str:
    return _hash_sources(Path(__file__).resolve().parent)


def _hash_sources(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or path.suffix not in _SOURCE_SUFFIXES
            or "__pycache__" in relative.parts
        ):
            continue
        digest.update(relative.as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()[:HASH_LENGTH]
