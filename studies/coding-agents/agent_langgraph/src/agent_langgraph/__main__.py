"""CLI:  python -m agent_langgraph --target NAME "task to perform"."""
from __future__ import annotations

import argparse
import os

from harness_common import prompts, targets, telemetry

from agent_langgraph.agent import build_agent, build_minimal_agent, build_smalltools_agent

# AGENT_LANGGRAPH_VARIANT selects the build: "full" deepagents (needs a strong
# model), "minimal" (FilesystemMiddleware), or "smalltools" (7B-friendly).
_VARIANTS = {
    "full": build_agent,
    "minimal": build_minimal_agent,
    "smalltools": build_smalltools_agent,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent_langgraph")
    parser.add_argument("--target", required=True, help="registered target name")
    parser.add_argument("--recursion-limit", type=int,
                        default=int(os.environ.get("AGENT_RECURSION_LIMIT", "50")))
    parser.add_argument("task", help="what the agent should do")
    args = parser.parse_args()

    target = targets.get_target(args.target)
    build = _VARIANTS.get(os.environ.get("AGENT_LANGGRAPH_VARIANT", "full"), build_agent)
    agent = build(target.path, test_cmd=list(target.test_cmd))
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompts.with_repo_listing(args.task, target.path)}]},
        config={"recursion_limit": args.recursion_limit},
    )
    print(result["messages"][-1].content)
    try:
        msgs = result.get("messages", [])
        turns = sum(1 for m in msgs if type(m).__name__ == "AIMessage")
        tokens = 0
        for m in msgs:
            meta = getattr(m, "usage_metadata", None) or {}
            tokens += int(meta.get("total_tokens", 0) or 0)
        telemetry.emit(turns, tokens or -1)
    except Exception:
        pass


if __name__ == "__main__":
    main()
