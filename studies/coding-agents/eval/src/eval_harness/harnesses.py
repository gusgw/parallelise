"""Registry of harnesses the eval can drive, and how to invoke each."""
from __future__ import annotations

from dataclasses import dataclass, field

from eval_harness.tasks import Task


@dataclass
class Harness:
    name: str
    project_dir: str   # directory under the harness root
    module: str        # python -m <module>
    env: dict = field(default_factory=dict)   # extra env (e.g. variant selector)


HARNESSES = {
    "agent_min": Harness("agent_min", "agent_min", "agent_min"),
    "agent_min_improved": Harness("agent_min_improved", "agent_min_improved", "agent_min_improved"),
    "agent_langgraph": Harness("agent_langgraph", "agent_langgraph", "agent_langgraph"),
    "agent_langgraph_small": Harness(
        "agent_langgraph_small", "agent_langgraph", "agent_langgraph",
        env={"AGENT_LANGGRAPH_VARIANT": "smalltools"}),
    "agent_pydantic": Harness(
        "agent_pydantic", "agent_pydantic", "agent_pydantic",
        env={"AGENT_PYDANTIC_VARIANT": "full"}),
    "agent_pydantic_min": Harness(
        "agent_pydantic_min", "agent_pydantic", "agent_pydantic",
        env={"AGENT_PYDANTIC_VARIANT": "min"}),
    "agent_pydantic_typed": Harness(
        "agent_pydantic_typed", "agent_pydantic", "agent_pydantic",
        env={"AGENT_PYDANTIC_VARIANT": "typed"}),
    "agent_pydantic_typed_checked": Harness(
        "agent_pydantic_typed_checked", "agent_pydantic", "agent_pydantic",
        env={"AGENT_PYDANTIC_VARIANT": "typed_checked"}),
}


def harness_command(harness: Harness, task: Task) -> list[str]:
    """Build the shell command to run a harness on a task (via uv, in its dir)."""
    return ["uv", "run", "python", "-m", harness.module,
            "--target", task.target, task.prompt]
