"""CLI:  python -m agent_pydantic --target NAME "task to perform".

AGENT_PYDANTIC_VARIANT selects the build: "full" (forcing, default), "min"
(no forcing), or "typed" (structured-output finish).
"""
from __future__ import annotations

import argparse
import os

from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from harness_common import prompts, targets, telemetry

from agent_pydantic.agent import (
    RunState,
    build_agent,
    build_min_agent,
    build_typed_agent,
    build_typed_checked_agent,
)

_VARIANTS = {
    "full": build_agent,
    "min": build_min_agent,
    "typed": build_typed_agent,
    "typed_checked": build_typed_checked_agent,
}
_DEPS_VARIANTS = {"full", "typed_checked"}    # variants that take RunState deps


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent_pydantic")
    parser.add_argument("--target", required=True, help="registered target name")
    parser.add_argument("--request-limit", type=int,
                        default=int(os.environ.get("AGENT_REQUEST_LIMIT", "40")))
    parser.add_argument("task", help="what the agent should do")
    args = parser.parse_args()

    variant = os.environ.get("AGENT_PYDANTIC_VARIANT", "full")
    build = _VARIANTS.get(variant, build_agent)

    target = targets.get_target(args.target)
    agent = build(target.path, test_cmd=list(target.test_cmd))
    prompt = prompts.with_repo_listing(args.task, target.path)

    kwargs = {"usage_limits": UsageLimits(request_limit=args.request_limit)}
    if variant in _DEPS_VARIANTS:
        kwargs["deps"] = RunState()
    try:
        result = agent.run_sync(prompt, **kwargs)
    except (UnexpectedModelBehavior, UsageLimitExceeded) as exc:
        # The model could not produce an acceptable final answer (e.g. exhausted its
        # output-retry budget). Record a clean failure rather than a traceback; any
        # edits it made are already on disk for the harness to test.
        print(f"agent did not finish cleanly: {type(exc).__name__}: {exc}")
        return
    print(result.output)
    try:
        u = result.usage()
        total = getattr(u, "total_tokens", None)
        if total is None:
            total = (getattr(u, "input_tokens", 0) or 0) + (getattr(u, "output_tokens", 0) or 0)
        telemetry.emit(getattr(u, "requests", -1), total or -1)
    except Exception:
        pass


if __name__ == "__main__":
    main()
