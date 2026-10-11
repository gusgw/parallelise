# FINDINGS — building effective deep agents on a 12B local model

What we learned driving three agent harnesses (hand-rolled loop, LangGraph +
deepagents, Pydantic AI) against a real failing-test task with **Gemma 4 12B**
(QAT W4A16, vLLM, 32k context). Every claim below is backed by a captured
conversation log or a source-level reading of the installed frameworks.

## The measurement

Task: `best_weather`'s test suite fails to collect (three stale names imported
from `pipeline.py`). The agent must find and fix the cause so the tests pass.
Best-of-N at sampling temperature 0.7, each attempt in an isolated git worktree
(`python -m eval_harness --task debug-best-weather --n 10 --concurrency 2`).

| harness | before tuning | after tuning (N=10) | median s |
|---|---|---|---|
| agent_min (hand-rolled loop)        | 1/3  | 1/10 | 409 |
| agent_langgraph (full deepagents)   | 0/3  | **3/10** | 411 |
| agent_langgraph_small (create_agent + our 6 tools) | — | **3/10** | 486 |
| agent_pydantic (Pydantic AI)        | 0/3  | 0/10 | 178 |

Headlines:

- **Configuration, not framework choice, dominated the outcome.** The full
  deepagents stack and the minimal create_agent variant score identically once
  both carry the same affordances (right tools, guards, finish gate, prompt).
- The frameworks went from *never editing a file* to fixing the suite 3/10 —
  every gain traceable to a specific, log-diagnosed defect.
- The 12B is the remaining ceiling: it reasons correctly (diagnoses the bug,
  plans the right edit) but fumbles mechanics (exact-string matching, argument
  schemas) often enough that even the best configuration lands ~30%.

## Failure-mode catalogue (all observed live, in order of discovery)

1. **Read-only agent.** The Pydantic challenger had only read/ls/grep but its
   prompt claimed it could edit and test. It "succeeded" at describing fixes in
   prose. Check the tool surface before blaming the model.
2. **Prose-finish loophole.** Both frameworks end a run on any response with no
   tool calls — narration counts as success. Pydantic AI closes it first-class
   (`@agent.output_validator` + per-step `tool_choice="required"` via callable
   `model_settings`); langchain needs a hand-rolled middleware (our
   `FinishGateMiddleware`: refuse to finish until an edit happened and tests
   reran after it).
3. **Swallowed tool errors.** Returning "ERROR: ..." strings from tools looks
   like success to the framework; Pydantic's retry machinery only engages if
   tools raise `ModelRetry`. Use the framework's error channel.
4. **Step-budget arithmetic.** deepagents' todo middleware adds a graph node, so
   one model-tool iteration costs 3 recursion steps, not 2 — `recursion_limit=50`
   is only ~16 turns, and todo bookkeeping burns ~1/3 of them on zero repo
   progress. Trim via a registered `HarnessProfile`
   (`excluded_middleware={TodoListMiddleware, "SummarizationMiddleware"}`,
   `general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False)`).
5. **Dead-weight middleware.** deepagents' summarizer triggers at 170k tokens —
   it can never fire inside a 32k context. Know your stack's defaults.
6. **Empty execute environment.** `LocalShellBackend` defaults to `env={}`;
   bare `pytest` resolved to the *system* pytest (which can never pass the
   target's suite). Pass `inherit_env=True` and put the exact test command in
   the prompt (`.venv/bin/python -m pytest -q`, relative, cwd = repo root).
7. **Prompt contradictions.** Our shared "paths are relative, never absolute"
   strategy directly contradicted the middleware's "all paths must start
   with /". Mechanically harmless (paths get normalized — a refuted hypothesis
   worth recording), but it costs a small model attention. The system prompt
   must agree with the framework's own conventions.
8. **Typography is load-bearing.** "grep for \`def \`" made the model grep for a
   literal backtick — 17 identical retries to recursion death. No backticks in
   prompts; give concrete, regex-safe patterns (pattern "def ").
9. **Read-view ≠ edit-view.** deepagents' read_file returns cat -n formatted
   text; its edit_file wants raw text. The 12B copied phantom indentation from
   the line-number gutter into old_string and could not see why edits failed —
   corrective error text did NOT help; it retried the same call. Fix the cause:
   we swapped in raw-text read_file/edit_file via `excluded_tools` + additive
   `tools=[...]`, and made edit errors *quote the file's real text*.
10. **Schema blending.** Under pressure the model passed edit_file's replacement
    as `content` (write_file's argument). Accept obvious aliases in your tools.
11. **Unbounded essays.** With loops fixed, the model ended one run with a
    46k-char prose answer (~7 min of generation). Cap generation
    (`max_tokens=3000`) — tool args fit, essays do not.
12. **No repeated-call guard anywhere.** Neither framework stops a model from
    re-issuing an identical failing call. Our `RepeatGuardMiddleware`
    (langchain) and a `ModelRetry`-raising guard (Pydantic) intercept exact
    repeats with corrective feedback. Note the limit observed in (9): a guard
    stops *waste* but cannot fix *misperception* — remove the cause instead.

## Design rules distilled

- **Audit the loop's affordances first**: can the agent actually edit, actually
  run the right tests, actually finish only when verified? Most of our "model
  failures" were harness failures.
- **Configure with the framework, not against it**: follow its path/prompt
  conventions; use its trim knobs; replace hostile built-in tools through its
  own extension points rather than fighting them in the prompt.
- **Enforce, don't advise**: snippet-size caps, tool_choice forcing, finish
  gates, and repeat guards beat prompt admonitions on a 12B every time.
- **Make errors teach**: a failed edit should quote the real file text; a
  validation failure should name the missing field. The model can only fix
  what it can see.
- **Account for steps and tokens**: know what one iteration costs against the
  recursion/request limit, and what each middleware adds to the context.
- **Measure warm**: temperature ~0.7 + best-of-N success rates; a single
  temperature-0 run is a coin you can only flip once.

## Phase 9 — the agent matrix at N=20 (frozen as `agents-frozen-v1`)

Seven agents, N=20 each on `debug-best-weather`, Gemma 4 12B @ 32k, temp 0.7,
10 h 17 m wall-clock. Primary metric = successful fixes (k/20); telemetry adds
median turns/tokens and a terminal-reason breakdown.

| agent | k/20 | med s | med turns | med ktok | dominant failure |
|---|---|---|---|---|---|
| agent_min (control) | 4 | 461 | — | — | 11× repeat-abort |
| agent_min_improved | 3 | 699 | — | — | 12× step-limit |
| agent_langgraph (deepagents) | 6 | 619 | — | — | 7× recursion-limit |
| agent_langgraph_small | 6 | 424 | — | — | 8× recursion-limit |
| agent_pydantic (full) | 6 | 707 | — | — | 6× tests-fail / 6× timeout |
| agent_pydantic_min | 4 | 602 | — | — | 9× tests-fail |
| agent_pydantic_typed | 0 | 40 | — | — | 12× no-edit (fabricated report) |

(turns/tokens telemetry added *after* this run; populated from Phase 10 on.)

Findings:
- **A band at 15–30 %, one clear collapse.** Six agents cluster at 3–6/20;
  Wilson 95 % CIs overlap, so the framework-vs-baseline gap (6 vs 4) is **not
  significant at N=20** — directionally consistent with the N=10 screen, not proof.
  The only clean separations: `typed` (0/20, ≤16 %) is genuinely worse, and the
  whole band clearly beats the temperature-0 era's all-loops failures.
- **Structured output is an escape hatch on a weak model.** `pydantic_typed`
  collapsed (median 40 s, 12× no-edit): forced to emit `FixReport(tests_pass=…)`,
  the 12B fabricates `tests_pass=True` with a zero-length diff on turn one. Never
  give a weak model a one-step structured "done" unless the claim is validated.
- **"Persist longer" ≠ better.** `agent_min_improved` (finish-gate + nudge) did
  **not** beat the control (3 vs 4) and was slower (median 699 s, 15/20 burned the
  full budget). The framework gains came from the tool/prompt fixes, not from
  refusing to stop early.
- **Validator-forcing helped Pydantic:** `full` (must-pass output validator) 6/20
  beat `min` (no forcing) 4/20 — it converted "finish-with-failing-tests" into more
  passes. `agent_pydantic` now has a real effect (tied best), the Phase-8 goal.
- **The timeout censors the slow agents** (3–6 each hit the 900 s cap, mostly at
  rc=1 — suite collects, one test fails); today's rates are lower bounds for them.
- **Token telemetry already pays off:** on a trivial task, `agent_min`/`pydantic_min`
  = 2 turns / 1.4k tok vs `agent_langgraph` = 6 turns / 17k tok — the deepagents
  middleware's ~10× context overhead, now quantified per run.

**Decision:** froze the 7 agents as-is (`agents-frozen-v1`) rather than tune
further on the slow 12B — the differences are noise-dominated at N=20, the defects
(`typed`, `min_improved`) are informative findings whose response to a stronger
model is itself a Phase-10 result, and the model is the higher-leverage variable.

## Phase 10 — model factorial (12B → 31B), `debug-best-weather`, N=20

Identical frozen matrix, two models. The model is the dominant lever:

| agent | 12B | 31B |
|---|---|---|
| `agent_langgraph` (deepagents) | 6/20 (30%) | 20/20 (100%) |
| `agent_langgraph_small` | 6/20 (30%) | 19/20 (95%) |
| `agent_pydantic` (full = prose + must-pass validator) | 6/20 (30%) | 18/20 (90%) |
| `agent_pydantic_min` | 4/20 (20%) | 16/20 (80%) |
| `agent_min_improved` | 3/20 (15%) | 14/20 (70%) |
| `agent_min` (control) | 4/20 (20%) | 10/20 (50%) |
| `agent_pydantic_typed` (structured finish) | 0/20 | 0/20 |

- **The model lifts the whole band ~3×** and separates the agents: at 12B everything
  is noise around 15–30% (overlapping CIs); at 31B the frameworks pull clearly ahead
  of the bare loop (langgraph 20/20 vs min 10/20, non-overlapping). Frameworks help —
  but only once the model is strong enough to use them.
- **`min_improved` reverses with model strength:** persistence (finish-gate + repeat
  nudge) *helps* a capable model (31B 14 > min 10) but *wastes budget* on an incapable
  one (12B 3 < min 4). The same mechanism is a feature or a liability depending on
  whether the model can eventually get there.
- **Timeout censoring vanished** on the 31B (0 timeouts vs 3–6 on the 12B) — the
  earlier slow-agent rates were lower bounds; the L40S bandwidth offsets the 31B size.
- **`typed` stays at exactly 0 on *both* models — model-independent.** That pointed to
  a structural cause, investigated next.

## Phase 10 follow-up — why `agent_pydantic_typed` is stuck at zero

`typed` finishes by emitting a structured `FixReport{summary, tests_pass}` (pydantic-ai
`ToolOutput`). It scored 0/20 on both models, but for two *different* reasons:

- **12B: it lies.** The model emits `tests_pass=True` over a still-broken suite (nothing
  checks the claim) and stops. (`no-edit`/`still-broken`.)
- **31B: it crashes.** Forced to answer through the output tool, the model emits a storm
  of empty/non-actionable turns; pydantic-ai counts each against a *cumulative* output-
  retry budget and raises `UnexpectedModelBehavior`. The "tests-fail" label was a masked
  crash (a partial edit left on disk).

I built `agent_pydantic_typed_checked` to try to rescue the idea — same structured finish,
but the report is only accepted once the tests *really* pass (deps + output validator, as
in `full`), plus a clean failure path instead of a traceback. It removed the lying and the
crashing but **stayed at 0/12 (31B), CI [0%, 24%]** — vs `full`'s 90%. Escalation, each step
verified live on the 31B:

1. **Verify the claim** → no more lies, no more crashes; still 0. The model does `full`'s
   *correct* fix (remap the stale test imports to the real functions) but leaves a couple
   of references behind and runs out of budget.
2. **Concrete feedback + "don't gut the tests, go edit" directive + budget 8→30** → improved
   the *behaviour* (correct-shaped edits), not the *outcome*; still 0/12. (Six of twelve
   attempts produced a byte-identical near-miss diff.)
3. **Trace the turn-by-turn** → the killer is an **empty-turn storm**: ~10 empty model
   responses per real action. The model's edits are correct and incremental; it just can't
   survive the storm. `report_fix=0` — it never even reaches the finish; it dies mid-fix.
4. **Budget 80 + request cap 200** → 2/4 still exhaust the 80 retries (~565 s), 2/4 hit the
   900 s wall. Raising the budget only trades retry-exhaustion for timeout. Still 0.
5. **Tool-free structured finish (`PromptedOutput`)** → the storm vanishes, but a new failure:
   the model fixates on emitting the report JSON and **never calls a tool** (`calls=0`). Still 0.

**Conclusion (multiply confirmed):** on this gemma4 build, forcing the final answer into a
structured schema — by *any* pydantic-ai mechanism (output tool *or* prompted JSON) — breaks
the agent's tool-use loop. The value was never the structured output; it is the **verification
gate** (validator on real test results), and that must be delivered through a **plain-prose
finish** (`build_agent`/`full`, 90%). `typed_checked` is retained as a known-zero control (budget
dialed back so it fails fast and cheap), to confirm the temperature sweep doesn't rescue it either.

## Phase 11 — temperature sweep (31B, `debug-best-weather`, N=20)

Full battery, `MODEL_TEMPERATURE` ∈ {0.0, 0.3, 0.7, 1.0} (0.7 reused from Phase 10):

| agent | T=0.0 | T=0.3 | T=0.7 | T=1.0 |
|---|---|---|---|---|
| `agent_langgraph` | 20/20 | 20/20 | 20/20 | 20/20 |
| `agent_langgraph_small` | 20/20 | 20/20 | 19/20 | 20/20 |
| `agent_pydantic` (full) | 6/20 | 12/20 | 18/20 | **20/20** |
| `agent_pydantic_min` | 7/20 | 10/20 | 16/20 | **20/20** |
| `agent_min_improved` | 0/20 | 5/20 | 14/20 | 15/20 |
| `agent_min` (control) | 0/20 | 4/20 | 10/20 | 13/20 |
| `agent_pydantic_typed` | 0/20 | 0/20 | 0/20 | 0/20 |
| `agent_pydantic_typed_checked` | 0/20 | 0/20 | (0/12) | 0/20 |

**The headline is counterintuitive: higher temperature is monotonically *better* here, and
greedy decoding (T=0.0) is catastrophic for the weakly-scaffolded agents.** This is the
opposite of the usual "lower temperature = more reliable" intuition, and it is large and
significant — e.g. `agent_min` 0/20 → 13/20 (CIs [0,16%] vs [43,82%]), `pydantic` (full)
6/20 → 20/20 ([15,52%] vs [84,100%]). Every mid-band agent climbs steadily with temperature.

- **Mechanism (confirms the `default_temperature` docstring's bet):** at T=0.0 the model is
  deterministic, so the first wrong move — a mis-targeted exact-string edit, a repeated tool
  call — recurs forever; the agent can never escape a stuck state. Sampling diversity *is* the
  escape hatch. The bare loops, which rely entirely on the model self-correcting, collapse to
  0/20 at T=0.0; the verification-gated pydantic agents keep ~30% (the validator forces re-tries
  that occasionally differ) but only reach their ceiling at T=1.0.
- **Scaffolding substitutes for temperature.** `agent_langgraph`/`_small` are flat at ~100%
  across the whole range — deepagents'/LangGraph's planning + retry structure recovers from
  stuck states without needing sampling to do it. The stronger the scaffold, the less the
  agent depends on temperature; the weaker the scaffold, the more temperature carries it.
- **Temperature does not move the structural zeros.** `typed`/`typed_checked` are 0/20 at every
  temperature — as predicted, the empty-turn storm / broken tool-loop is a structural defect,
  not a sampling one. This is the known-zero control doing its job.
- **No speed penalty:** median per-attempt ~110–150 s and **zero timeouts at every temperature**;
  T=0.0 is not meaningfully faster despite being greedy (its failures loop to the request limit).

**Design rule:** on a medium local model, do *not* reach for low temperature to get
determinism — for any agent that leans on the model to self-correct, it removes the only escape
from a loop. Pair weak scaffolding with high temperature (~1.0), or invest in scaffolding that
recovers structurally (LangGraph/deepagents) and the temperature knob stops mattering.

## Phase 12 — harder tasks: the solvable-but-hard band is (nearly) empty (31B)

Goal: find a task harder than the mechanical `debug-best-weather` (where temperature
is monotonic and strong agents sit at 100%) — one in the band where strong agents are
~20–80%, to see whether a harder task shows a temperature *optimum*. Three tasks were
designed (workflow-vetted), each built on its own labelled branch in the target repo
with a **frozen** behavioural spec (`check_frozen_test` restores the test from HEAD
before scoring, so deleting/weakening it can't pass). Each verified solvable by a
reference fix before the agents saw it.

| task | target | kind | result | why |
|---|---|---|---|---|
| `feature-inline-threshold` | mealworm | feature (multi-file) | **0/96** frozen budgets, **0/96** relaxed | can't coordinate the publish+consume change |
| `debug-s3-retry` | weather | debug (retry pattern) | **~100%** (all agents 2/2 smoke) | textbook pattern — too easy |
| `debug-msgid-trace` | mealworm | debug (subtle 1-file) | **0/24** (default + relaxed + guided) | can't localize; even pointed at it, can't fix |

- **#1 is capability-, not budget-walled.** First sweep (frozen budgets, N=8×4 temps)
  failed on step/recursion/usage limits. Relaxing the budgets (env knobs, 3×) flipped
  the failures to `tests-fail` — the agents now *finish* but produce wrong fixes — and
  the score stayed **0/96 at every temperature**. More budget ≠ more capability.
- **#3 is subtle in two ways.** At default/relaxed budgets the agents mostly `no-edit`
  (can't find the bug — `Message()` silently regenerating `message_id`). With the prompt
  pointing straight at the consume-step rebuild, they edit and run tests but still can't
  produce the correct fix (`tests-fail`). 0/24 across all conditions.
- **The band is essentially empty for this model.** Autonomous debugging on the 31B is
  bimodal: **mechanical / pattern tasks are trivial** (retry; the stale-import remap),
  **reasoning tasks are impossible** (multi-file coordination, localizing a subtle
  semantic bug, or fixing one even when handed its location). There is little middle
  ground, which is exactly why only the mechanical `debug-best-weather` yields a
  temperature gradient — the temperature×difficulty optimum can't be probed because no
  constructed task lands between trivial and impossible.
- **Harness additions:** `check_frozen_test` (tamper-proof scoring); agent budgets are
  env-overridable (`AGENT_MAX_STEPS`/`AGENT_RECURSION_LIMIT`/`AGENT_REQUEST_LIMIT`,
  defaults unchanged). Tasks live in gitignored `tasks_local.py` on per-task branches
  (`harness-task-inline` / `-s3retry` / `-msgid`).

## Open questions / next

- **A genuinely middle task** for the temperature optimum: the band is so narrow that
  hitting it likely needs a *mechanical* bug with a non-obvious *location* (search-hard,
  fix-easy) rather than a reasoning-hard fix — worth one more design pass.
- **Serving speed:** `--max-num-seqs 4` is now proven safe on the 12B (other agent: 24
  concurrent multi-tool requests, tool-call args intact; per-request parser, no spec
  decoding) for ~3.9× throughput. Scored 31B runs were kept at concurrency 2 for
  comparability with the existing baselines; adopt concurrency 4 after the identical
  argument-integrity check passes on the 31B.
