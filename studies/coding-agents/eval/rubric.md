# Eval rubric

How we judge a harness on each task. `run_eval` automates what it can; the rest is
manual review of the agent's diff and output.

## Automatic (scored by `run_eval`)
- **Result** — `PASS` / `FAIL` / `skip`:
  - `debug`, `test`, `optimise` tasks → the target's test suite passes afterwards
    (`check_tests_pass`).
  - `document`, `review` tasks → the expected artefact was written
    (`check_file_contains`); quality is judged manually (see below). These show as
    `skip` only if no check is attached.
- **Wall-clock seconds** — per (harness, task).

## Manual (reviewer fills in, per harness × task)
- **Correctness** — did it actually solve the task, not just make tests pass by
  deleting them? (Read the diff.)
- **Edit minimality** — smallest sensible change; no unrelated churn.
- **Tool-call behaviour** — did it use tools (vs narrate)? how many calls / retries
  / failures? (The narration failure mode is the key 7B risk — see
  `../common/MODEL_NOTES.md`.)
- **Output quality** — for `document`/`review`, is the prose accurate and useful?
- **Token cost** — if/when the model endpoint reports usage.

## Running it
```bash
# all harnesses over all tasks (needs targets present + a reachable model):
cd eval && python -m eval_harness --out RESULTS.md

# one harness, one task:
python -m eval_harness --harness agent_min --task <task>
```

## Status
The runner, tasks, checks, and rendering are built and unit-tested. A real scoring
run is **pending** the target codebases (targets + their deps) and ideally the
Gemma 4 endpoint; results will be written to `eval/RESULTS.md`.

`agent_min`, the LangGraph small-tools agent, and the Pydantic AI agent already run
end-to-end on Mistral-7B (see `../RESULTS.md`); the deepagents default stack needs
the stronger model.
