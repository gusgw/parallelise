"""Facts about the software and the machine, noted in every run record.

A result is only as useful as the account of what produced it: a hash of the package's source
says exactly which code ran, and the count of processor cores the run could use bears on every
speed-up it measures.
"""

from __future__ import annotations

import hashlib
import os
from functools import cache
from pathlib import Path


def cores() -> int:
    """Return the number of processor cores this process may run on, and at least 1.

    Only the processors the process is allowed to use count, which is what limits a benchmark in
    a container or under a command that pins processors; and a core counts once however many
    hardware threads it has, because a routine with a worker on every hardware thread waits at
    each step whenever any other program runs. Where the system does not say which core a
    processor belongs to, every processor counts.
    """
    try:
        allowed = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)
    found = set()
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            found.add((topology.joinpath("physical_package_id").read_text(),
                       topology.joinpath("core_id").read_text()))
        except OSError:
            return max(1, len(allowed))
    return max(1, len(found))


PACKAGE = Path(__file__).resolve().parent


@cache
def source_hash(root: Path = PACKAGE) -> str:
    """Return a short hash of the package's Python files, their names and their contents.

    It changes whenever a prompt, a template or the judgement changes, and when a file is
    added, removed or renamed.
    """
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" not in path.parts:
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()[:12]
