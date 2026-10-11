# agent_min — run notes

> **About this file.** This is a set of run notes, kept as a working record and not edited into
> finished prose. It uses abbreviations and a terse note style. The readable account of the
> study is [../README.md](../README.md). Abbreviations: N is the number of attempts per cell; CI is a 95% confidence interval; QAT is
> quantisation-aware training; W4A16 means 4-bit weights with 16-bit activations; T is the
> sampling temperature; B means billion parameters.

## Status
Minimal hand-rolled agent loop (httpx-only, no framework). Unit tests green
(`bash check.sh`, 9 tests). Validated **live end-to-end** against the Mistral
endpoint on a throwaway repo: task *"Use the read_file tool to read
note.txt and report the secret code"* → correctly returns "The secret code is
4242" after one real tool call.

## Live behaviour on Mistral-7B (2026-06-08)
- Loop mechanics are solid: the model emits `tool_calls`, we send results back (with
  9-char ids — see `common/MODEL_NOTES.md`), and the model produces a final answer.
- Tool-use **decisioning** is the weak point:
  - Vague task ("Read note.txt and tell me the code") → the model does NOT call
    the tool; it hallucinates a code or tells the user to run `cat`.
  - Explicit task ("Use the read_file tool …") + **short** system prompt → calls
    the tool and reports the real content. `DEFAULT_SYSTEM` is tuned for this.
  - A long/forceful "you MUST call tools" prompt **backfires**: the model narrates
    "I have used the read_file tool …" without actually calling it.

## Not yet done
- Phase 0.3 against `best_weather` (the failing-test fix) needs the target's deps
  installed (faststream and others), absent on the development machine. Run it
  after `uv sync` in the target.

## How to run
    . ../common/env.sh
    PYTHONPATH=../common/src:src python -m agent_min --target best_weather \
        "Use the read_file/grep/run_tests tools to find and fix the failing test."

## Temperature & loop guard (Phase 8, 2026-06-10)
- Sampling temperature now comes from `MODEL_TEMPERATURE` (default 0.7) via
  `harness_common.model.default_temperature()`. `run_agent` calls `chat()` without
  forcing a temperature, so the agent runs *warm* and can escape a stuck action. At
  temperature 0 it deterministically repeats a failing call — see the temperature
  section of the study README.
- `loop.py` keeps a **repeated-identical-tool-call guard**: the same call twice in a
  row → stop ("repeated the same tool call without progress").
- The run is seeded with the repo's top-level listing and a shared debugging STRATEGY
  (`harness_common.prompts`), used verbatim by all three agents for a fair comparison.
