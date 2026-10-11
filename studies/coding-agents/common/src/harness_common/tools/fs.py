"""Filesystem tools confined to a repo root (reject .. and absolute escapes)."""
from __future__ import annotations

import re
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache", ".ruff_cache"}


def safe_path(root: str, path: str) -> Path:
    """Resolve path under root; raise ValueError if it escapes root."""
    root_p = Path(root).resolve()
    target = (root_p / path).resolve()
    if target != root_p and root_p not in target.parents:
        raise ValueError("path escapes root: " + path)
    return target


# Sized for a >=32k-token context: large enough to read a whole module (e.g. a
# ~575-line file) so definitions aren't hidden, while still bounding huge files.
DEFAULT_READ_MAX_CHARS = 24000


def read_file(root: str, path: str, start: int = 1, count: int | None = None,
              max_chars: int = DEFAULT_READ_MAX_CHARS) -> str:
    """Read a file under root, or lines start..start+count-1; truncate at max_chars.

    Large files are truncated with a notice so the agent's context stays small:
    use grep to locate code, then read a line range with (start, count).
    """
    text = safe_path(root, path).read_text()
    if count is not None:
        lines = text.splitlines()
        chosen = lines[max(start - 1, 0): start - 1 + count]
        prefix = "[lines " + str(start) + "-" + str(start + len(chosen) - 1) + " of " + str(len(lines)) + "]\n"
        body = "\n".join(chosen)
    else:
        prefix = ""
        body = text
    if len(body) > max_chars:
        body = body[:max_chars] + "\n[truncated: " + str(len(body)) + " chars total; read a line range (start, count) or use grep]"
    return prefix + body


def write_file(root: str, path: str, content: str) -> str:
    """Write text to a file under root (creating parents); return a short summary."""
    target = safe_path(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return "wrote " + str(len(content)) + " chars to " + path


def nearest_region(text: str, old: str, context: int = 2) -> str:
    """Quote the file text around the first line of old, so a failed edit's
    error shows what the file really contains (models mis-copy indentation)."""
    probe = next((line.strip() for line in old.splitlines() if line.strip()), "")
    if not probe:
        return ""
    lines = text.splitlines()
    hit = next((i for i, line in enumerate(lines) if probe in line), None)
    if hit is None:
        return ""
    lo = max(0, hit - context)
    hi = min(len(lines), hit + max(4, old.count("\n") + 3))
    return "\nThe file actually contains (copy old exactly from this):\n" + "\n".join(lines[lo:hi])


def edit_file(root: str, path: str, old: str, new: str) -> str:
    """Replace every occurrence of old with new in a file under root."""
    target = safe_path(root, path)
    text = target.read_text()
    count = text.count(old)
    if count == 0:
        raise ValueError("old text not found in " + path + nearest_region(text, old))
    target.write_text(text.replace(old, new))
    return "replaced " + str(count) + " occurrence(s) in " + path


def ls(root: str, path: str = ".") -> list[str]:
    """List names in a directory under root; directories get a trailing slash."""
    target = safe_path(root, path)
    names = []
    for entry in sorted(target.iterdir()):
        names.append(entry.name + "/" if entry.is_dir() else entry.name)
    return names


def glob(root: str, pattern: str) -> list[str]:
    """Return paths (relative to root) matching a glob pattern, recursively."""
    root_p = Path(root).resolve()
    return sorted(str(p.relative_to(root_p)) for p in root_p.glob(pattern))


def grep(root: str, pattern: str, path: str = ".") -> list[str]:
    """Search text files under path for a regex; return 'relpath:lineno:line' hits."""
    root_p = Path(root).resolve()
    base = safe_path(root, path)
    rx = re.compile(pattern)
    hits = []
    files = sorted(base.rglob("*")) if base.is_dir() else [base]
    for f in files:
        if not f.is_file():
            continue
        rel = f.relative_to(root_p)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        try:
            text = f.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if rx.search(line):
                hits.append(str(rel) + ":" + str(i) + ":" + line.strip())
    return hits
