"""nano_pydantic — a minimal Pydantic-AI coding agent with a verification gate.

Self-contained: needs only `pydantic-ai` (see requirements.txt). It drives the same
read / write / run_tests tools as `nano/`, but through Pydantic-AI, and adds the one
thing that made Pydantic effective in testing: an output validator that refuses to
finish until the tests actually pass.

    MODEL_BASE_URL=http://localhost:4000/v1 MODEL_NAME=your-model \
        python pyd_agent.py /path/to/repo "make the tests pass"

Laid out in build order: (1) the model, (2) per-run state, (3) the agent — tools
plus the gate.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits


# --- 1. the model ----------------------------------------------------------
def build_model() -> OpenAIChatModel:
    base = os.environ.get("MODEL_BASE_URL", "http://localhost:8000/v1")
    name = os.environ.get("MODEL_NAME", "local")
    key = os.environ.get("MODEL_API_KEY") or "not-needed"
    return OpenAIChatModel(name, provider=OpenAIProvider(base_url=base, api_key=key))


# --- 2. per-run state: have the tests passed since the last edit? -----------
@dataclass
class State:
    tests_passed: bool = False


# --- 3. the agent: three tools + a gate that won't finish on a red suite ----
def build_agent(root, *, test_cmd=("python", "-m", "pytest", "-q")):
    agent = Agent(
        build_model(),
        deps_type=State,
        retries=4,
        model_settings=ModelSettings(
            temperature=float(os.environ.get("MODEL_TEMPERATURE", "0.7"))),
        instructions=(
            "You are a coding agent. Use the tools to read and edit files and run "
            "the tests. Make small, exact edits and copy text exactly from "
            "read_file. Run run_tests to check your work; only finish once it "
            "reports success."),
    )

    @agent.tool_plain
    def read_file(path: str) -> str:
        """Read a text file in the repo."""
        with open(os.path.join(root, path), encoding="utf-8") as handle:
            return handle.read()

    @agent.tool
    def write_file(ctx: RunContext[State], path: str, content: str) -> str:
        """Create or overwrite a repo file."""
        with open(os.path.join(root, path), "w", encoding="utf-8") as handle:
            handle.write(content)
        ctx.deps.tests_passed = False     # a new edit invalidates the last test run
        return "wrote " + str(len(content)) + " chars to " + path

    @agent.tool
    def run_tests(ctx: RunContext[State]) -> str:
        """Run the repo's pytest suite."""
        proc = subprocess.run(list(test_cmd), cwd=root, capture_output=True,
                              text=True, timeout=600)
        ctx.deps.tests_passed = proc.returncode == 0
        return "exit " + str(proc.returncode) + "\n" + (proc.stdout + proc.stderr)[-3000:]

    @agent.output_validator
    def must_pass(ctx: RunContext[State], output: str) -> str:
        # the verification gate: do not accept a final answer until run_tests is green
        if not ctx.deps.tests_passed:
            raise ModelRetry("The tests are not passing yet. Keep editing and rerun "
                             "run_tests; only finish once it reports success.")
        return output

    return agent


def main():
    if len(sys.argv) < 3:
        print("usage: python pyd_agent.py <repo-dir> <task>")
        raise SystemExit(2)
    agent = build_agent(sys.argv[1])
    result = agent.run_sync(sys.argv[2], deps=State(),
                            usage_limits=UsageLimits(request_limit=40))
    print(result.output)


if __name__ == "__main__":
    main()
