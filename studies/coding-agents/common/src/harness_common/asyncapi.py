"""Read a FastStream pipeline's AsyncAPI schema and render a compact topology.

parse_topology / render_topology are pure (unit-tested on a sample dict).
generate_asyncapi shells out to the faststream CLI (needs faststream + the app
importable) and is exercised against real targets in Phase 1.4.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from harness_common.tools import proc


@dataclass
class Topology:
    channels: list[str] = field(default_factory=list)          # topic addresses
    operations: list[tuple[str, str]] = field(default_factory=list)  # (action, address)


def parse_topology(spec: dict) -> Topology:
    """Extract channel addresses and operations from an AsyncAPI 3.0 dict."""
    key_to_addr = {}
    for key, ch in spec.get("channels", {}).items():
        key_to_addr[key] = ch.get("address", key)
    topo = Topology(channels=sorted(set(key_to_addr.values())))
    for op in spec.get("operations", {}).values():
        action = op.get("action", "?")
        ref = op.get("channel", {}).get("$ref", "")
        ch_key = ref.split("/")[-1]
        topo.operations.append((action, key_to_addr.get(ch_key, ch_key)))
    topo.operations.sort()
    return topo


def render_topology(topo: Topology) -> str:
    """Render a compact, LLM-friendly summary of the topology."""
    verbs = {"receive": "subscribes", "send": "publishes"}
    lines = ["channels (topics): " + ", ".join(topo.channels)]
    for action, addr in topo.operations:
        lines.append("  " + verbs.get(action, action) + " " + addr)
    return "\n".join(lines)


def generate_asyncapi(target_path: str, app: str, timeout: float = 120.0) -> dict:
    """Run `faststream docs gen <app>` in target_path and return the parsed schema."""
    out = Path(target_path) / "asyncapi.json"
    pre_existing = out.exists()
    result = proc.run(["faststream", "docs", "gen", app], target_path, timeout)
    if not out.exists():
        raise RuntimeError("faststream docs gen produced no asyncapi.json:\n" + result.output)
    try:
        spec = json.loads(out.read_text())
    finally:
        if not pre_existing:
            out.unlink()
    return spec
