"""Uniform per-run telemetry line that agents emit and the eval parses.

An agent prints exactly one `TELEMETRY {...}` line at the end of its run; the eval
reads it back from the agent's stdout. Emitting is best-effort and must never
affect the agent's behaviour, so callers wrap emit() in try/except.
"""
from __future__ import annotations

import json

PREFIX = "TELEMETRY "


def emit(turns: int, tokens: int) -> None:
    """Print the telemetry line (turns = model round-trips, tokens = total)."""
    print(PREFIX + json.dumps({"turns": turns, "tokens": tokens}))


def parse(text: str) -> dict:
    """Return the last telemetry dict found in text, or {} if none/garbled."""
    found: dict = {}
    for line in (text or "").splitlines():
        if line.startswith(PREFIX):
            try:
                found = json.loads(line[len(PREFIX):])
            except Exception:
                pass
    return found
