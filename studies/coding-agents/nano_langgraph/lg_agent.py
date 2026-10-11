"""nano_langgraph — minimal LangGraph coding agents, in two flavours.

Self-contained (see requirements.txt). Two bare-bones agents over the same model:

  * build_deep()  — deepagents.create_deep_agent. Filesystem tools and a shell
    `execute` come built in from the backend, so you give it only a model and a
    prompt. This is the "deepagents" reference, stripped to its smallest form.

  * build_plain() — langchain.agents.create_agent + three plain tools (read /
    write / run_tests). No deepagents; lighter to install and fewer moving parts.

Run either flavour:

    MODEL_BASE_URL=http://localhost:4000/v1 MODEL_NAME=your-model \
        python lg_agent.py [deep|plain] /path/to/repo "make the tests pass"
"""
from __future__ import annotations

import os
import subprocess
import sys


# --- the model (shared by both flavours) -----------------------------------
def build_model():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url=os.environ.get("MODEL_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("MODEL_API_KEY") or "not-needed",
        model=os.environ.get("MODEL_NAME", "local"),
        temperature=float(os.environ.get("MODEL_TEMPERATURE", "0.7")),
    )


SYSTEM = (
    "You are a coding agent. Read and edit files and run the tests using your "
    "tools. Make small, exact edits and copy text exactly from what you read. Run "
    "the tests to check your work and only finish once they pass."
)


# --- flavour 1: deepagents (built-in filesystem tools + a shell `execute`) --
def build_deep(root):
    from deepagents import create_deep_agent
    from deepagents.backends import LocalShellBackend
    return create_deep_agent(
        model=build_model(),
        system_prompt=SYSTEM + ' Run the tests with execute(command="python -m pytest -q").',
        backend=LocalShellBackend(root_dir=root, virtual_mode=True, inherit_env=True),
    )


# --- flavour 2: plain LangGraph (create_agent + three plain tools) ----------
def build_plain(root, test_cmd=("python", "-m", "pytest", "-q")):
    from langchain.agents import create_agent

    def read_file(path: str) -> str:
        """Read a text file in the repo."""
        with open(os.path.join(root, path), encoding="utf-8") as handle:
            return handle.read()

    def write_file(path: str, content: str) -> str:
        """Create or overwrite a repo file."""
        with open(os.path.join(root, path), "w", encoding="utf-8") as handle:
            handle.write(content)
        return "wrote " + str(len(content)) + " chars to " + path

    def run_tests() -> str:
        """Run the repo's pytest suite."""
        proc = subprocess.run(list(test_cmd), cwd=root, capture_output=True,
                              text=True, timeout=600)
        return "exit " + str(proc.returncode) + "\n" + (proc.stdout + proc.stderr)[-3000:]

    return create_agent(model=build_model(), system_prompt=SYSTEM,
                        tools=[read_file, write_file, run_tests])


def run(agent, task):
    result = agent.invoke({"messages": [{"role": "user", "content": task}]},
                          config={"recursion_limit": 50})
    return result["messages"][-1].content


def main():
    if len(sys.argv) < 4:
        print("usage: python lg_agent.py [deep|plain] <repo-dir> <task>")
        raise SystemExit(2)
    flavour, root, task = sys.argv[1], sys.argv[2], sys.argv[3]
    agent = build_deep(root) if flavour == "deep" else build_plain(root)
    print(run(agent, task))


if __name__ == "__main__":
    main()
