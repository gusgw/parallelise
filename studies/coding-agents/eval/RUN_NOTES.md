# eval — run notes

> **About this file.** This is a set of run notes, kept as a working record and not edited into
> finished prose. It uses abbreviations and a terse note style. The readable account of the
> study is [../README.md](../README.md). Abbreviations: N is the number of attempts per cell; CI is a 95% confidence interval; QAT is
> quantisation-aware training; W4A16 means 4-bit weights with 16-bit activations; T is the
> sampling temperature; B means billion parameters.

## What it is
`eval_harness` runs each agent harness (agent_min / agent_langgraph / agent_pydantic)
over a fixed task set against the real target pipelines, and scores the result.
The pure-logic parts are unit-tested with fakes (no agent, no model, no targets), so
`bash check.sh` is fast and pinpoints typos.

## Layout (small, single-purpose modules)
- `tasks.py`     — `Task` dataclass + `TASKS` (debug / test / document / review / optimise);
                   each carries a prompt, target, reset, and an automatic `check` where possible.
- `harnesses.py` — `Harness` dataclass + `HARNESSES`; `harness_command()` builds the
                   `uv run python -m <agent> --target <t> "<prompt>"` argv.
- `runner.py`    — single-shot (`run_one` / `run_eval` / `render_markdown`) and
                   best-of-N (`run_attempt` / `best_of_n` / `render_bestofn`).
- `worktree.py`  — per-attempt git worktree isolation (see below).
- `__main__.py`  — the CLI.

## How to run
Single shot (resets the target in place):

    . ../common/env.sh
    uv run python -m eval_harness --task debug-best-weather --out RESULTS.md

Best-of-N (Phase 8 — isolated worktrees, sampling at temperature):

    MODEL_TEMPERATURE=0.7 uv run python -m eval_harness \
        --task debug-best-weather --n 5 --concurrency 2 --out RESULTS.md

- `--n N`           : run each (harness, task) N times; report success rate k/N + median wall-clock.
- `--concurrency C` : up to C attempts at once. The model is single-slot, so this overlaps the
                      local work (pytest, file and git operations), not the inference itself.

## Worktree isolation (why best-of-N is safe to parallelise)
Each attempt runs in its own **detached git worktree of the target at HEAD**
(`worktree.add_worktree`): a fresh copy of the failing-test state, with the target's
`.venv` symlinked in so its test command still works. The agent's edits and test runs
happen *in the worktree*; the real working tree is never touched (uncommitted files
stay put). The worktree is removed afterwards. The agent is pointed at the worktree by
the `HARNESS_TARGET_PATH` env override, which `harness_common.targets.get_target`
honours. Concurrent git-metadata mutations are serialised by a lock; the slow
agent/test work still overlaps.

## Temperature
All three agents read `MODEL_TEMPERATURE` (default 0.7) via
`harness_common.model.default_temperature()`. Best-of-N samples this warm ensemble;
the success rate k/N is the honest metric, because a single temperature-0 shot is
deterministic and can lock into a loop (statistical-mechanics rationale in the
temperature section of the study README).

## Harness variants
`Harness` carries an optional `env` dict merged into each attempt's environment;
`agent_langgraph_small` selects the smalltools variant via
`AGENT_LANGGRAPH_VARIANT=smalltools`, giving a fourth comparison row (agent_min's
six tools through LangGraph's create_agent — isolates framework cost).

## N=10 result (Gemma 4 12B @ 32k, temperature 0.7, 2026-06-12)
agent_min 1/10 (median 409s) | agent_langgraph 3/10 (411s) |
agent_langgraph_small 3/10 (486s) | agent_pydantic 0/10 (178s, rc=1 fast fails).
Full analysis and the failure-mode catalogue: top-level FINDINGS.md.

## Tests (typo isolation)
`bash check.sh` = py_compile (syntax) → import-all smoke (walks `eval_harness`, so a
new module with a typo fails immediately) → pytest. Best-of-N is tested with fakes for
the runner, worktree, and clock (`test_runner.py`); `test_worktree.py` exercises a real
temp git repo. No model or targets needed.
