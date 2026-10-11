"""Small prompt helpers shared by the agents."""
from __future__ import annotations

from harness_common.tools import fs

# Shared debugging strategy, used verbatim by every agent's system prompt so the
# comparison is fair. Tuned for Gemma 4 12B, which needs the strategy spelled out.
# No backticks: a small model copies quote characters into tool arguments.
STRATEGY = (
    "All tool paths are relative to the repository root (e.g. test_pipeline.py) "
    "— never prefix a path with the project name or an absolute path. Files can be "
    "large: use grep to locate code, then read a line range. To fix a failing "
    "import or missing name, grep the source with pattern \"def \" to list the "
    "functions that really exist, compare them to what the test imports, then "
    "call edit_file. edit_file replaces one short snippet: pass a few exact lines "
    "as old, never a whole file. Act on what you find — do not repeat a call that "
    "already returned its result."
)


def with_repo_listing(task: str, root: str) -> str:
    """Append the repo's top-level file listing so the model doesn't guess paths."""
    files = "\n".join(fs.ls(root, "."))
    return task + "\n\nRepository files (paths are relative to here):\n" + files
