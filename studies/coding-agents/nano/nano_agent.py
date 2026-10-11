"""nano_agent — the smallest useful coding agent: an httpx tool-calling loop.

This file is self-contained: it needs only `httpx` and the standard library,
and nothing else in this repository, so it is a working agent on its own. Point it at an OpenAI-compatible chat endpoint:

    MODEL_BASE_URL=http://localhost:4000/v1 MODEL_NAME=your-model \
        python nano_agent.py /path/to/repo "make the tests pass"

It reads, writes, and runs the tests in one directory, looping until the model
says it is done or the step budget runs out.

The file is laid out in the order you build it: (1) the model call, (2) the
tools, (3) the loop.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import httpx


# --- 1. the model call -----------------------------------------------------
# One POST to /chat/completions. Returns the assistant message (which may carry
# tool_calls). Everything else is built on this one function.

def chat(messages, tools):
    base = os.environ.get("MODEL_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    name = os.environ.get("MODEL_NAME", "local")
    key = os.environ.get("MODEL_API_KEY", "")
    headers = {"Authorization": "Bearer " + key} if key else {}
    payload = {
        "model": name,
        "messages": messages,
        "tools": tools,
        "temperature": float(os.environ.get("MODEL_TEMPERATURE", "0.7")),
    }
    resp = httpx.post(base + "/chat/completions", json=payload,
                      headers=headers, timeout=600.0)
    resp.raise_for_status()
    msg = resp.json()["choices"][0]["message"]
    if msg.get("content") is None:        # some servers reject a null on resend
        msg["content"] = ""
    return msg


# --- 2. the tools ----------------------------------------------------------
# Three tools are enough to fix code: read a file, write a file, run the tests.
# Each takes the repo root so the agent cannot wander outside it.

def read_file(root, path):
    with open(os.path.join(root, path), encoding="utf-8") as handle:
        return handle.read()


def write_file(root, path, content):
    with open(os.path.join(root, path), "w", encoding="utf-8") as handle:
        handle.write(content)
    return "wrote " + str(len(content)) + " chars to " + path


def run_tests(root):
    proc = subprocess.run(["python", "-m", "pytest", "-q"], cwd=root,
                          capture_output=True, text=True, timeout=600)
    return "exit " + str(proc.returncode) + "\n" + (proc.stdout + proc.stderr)[-3000:]


# The schema the model sees. These names must match the functions above and the
# dispatch below — a mismatch is the classic typo, so a test checks it.
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file in the repo.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file", "description": "Create or overwrite a repo file.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "run_tests", "description": "Run the repo's pytest suite.",
        "parameters": {"type": "object", "properties": {}}}},
]


def dispatch(root, name, args):
    if name == "read_file":
        return read_file(root, args["path"])
    if name == "write_file":
        return write_file(root, args["path"], args["content"])
    if name == "run_tests":
        return run_tests(root)
    return "unknown tool: " + name


# --- 3. the loop -----------------------------------------------------------
SYSTEM = (
    "You are a coding agent. Use the tools to read and edit files and run the "
    "tests. Make small, exact edits. read_file shows the file as it is — copy "
    "text from it exactly. Run run_tests to check your work, and only stop once "
    "the tests pass. Never describe a change without making it."
)


def run_agent(task, root, max_steps=20):
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": task}]
    for _ in range(max_steps):
        msg = chat(messages, TOOLS)
        messages.append(msg)
        calls = msg.get("tool_calls")
        if not calls:
            return msg.get("content") or "(no answer)"
        for call in calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
                result = dispatch(root, name, args)
            except Exception as exc:
                result = "error: " + str(exc)
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": str(result)})
    return "(stopped: reached max_steps)"


def main():
    if len(sys.argv) < 3:
        print("usage: python nano_agent.py <repo-dir> <task>")
        raise SystemExit(2)
    print(run_agent(sys.argv[2], sys.argv[1]))


if __name__ == "__main__":
    main()
