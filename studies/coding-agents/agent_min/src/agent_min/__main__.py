"""CLI:  python -m agent_min --target NAME "task to perform"."""
from __future__ import annotations

import argparse
import os

from harness_common import targets, telemetry

from agent_min.loop import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent_min")
    parser.add_argument("--target", required=True, help="registered target name")
    parser.add_argument("--max-steps", type=int,
                        default=int(os.environ.get("AGENT_MAX_STEPS", "20")))
    parser.add_argument("task", help="what the agent should do")
    args = parser.parse_args()

    target = targets.get_target(args.target)
    usage: dict = {}
    result = run_agent(
        args.task, target.path, test_cmd=target.test_cmd, max_steps=args.max_steps, usage=usage
    )
    print(result)
    try:
        telemetry.emit(usage.get("calls", -1), usage.get("total_tokens", -1))
    except Exception:
        pass


if __name__ == "__main__":
    main()
