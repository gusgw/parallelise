# Agent performance study — full report

> **About this file.** This is a long working report, kept as a working record and not edited into
> finished prose. It uses abbreviations and a terse note style. The readable account of the
> study is [README.md](README.md). Abbreviations: N is the number of attempts per cell; CI is a 95% confidence interval; QAT is
> quantisation-aware training; W4A16 means 4-bit weights with 16-bit activations; T is the
> sampling temperature; B means billion parameters.

*A study of how to build an effective autonomous coding agent on a medium-size,
self-hosted LLM, by comparing several agent designs head-to-head on real pipeline
code under a controlled, one-variable-at-a-time protocol.*

Pairs with `FINDINGS.md` (the running log of the work). All numbers below are taken from the per-attempt
artifacts (`RESULTS*.jsonl`).

---

## 1. TL;DR

1. **On this task, the model was the biggest lever.** Holding the agents and task fixed and
   swapping the 12B model for the 31B model lifted the whole field about threefold (a band
   of 15% to 30% became 50% to 100%) and, for the first time, separated the agents from each other. On a weak
   model the agent design barely matters; on a capable one it matters a lot.
2. **Frameworks beat the bare loop — but only once the model is strong enough to use
   them.** At 12B everything is noise (~15–30%, overlapping intervals). At 31B the
   framework agents (LangGraph/deepagents 100%, Pydantic-AI 90%) pull clearly ahead of
   the hand-rolled loop (50%).
3. **On this task, with these two models, higher sampling temperature was better, not
   worse**, which is the opposite of the usual expectation. Greedy decoding (temperature 0)
   gave very poor results for the weakly scaffolded agents, and for the agents where the
   temperature mattered at all, success rose with temperature up to 1.0. Strong
   scaffolding made the setting irrelevant.
4. **Forcing the final answer into a structured object breaks the agent.** Every
   "structured finish" variant scored 0; the value is in *verification*, which must be
   delivered through a plain-prose finish.
5. **The model's capability is bimodal.** It solves *mechanical / pattern* tasks
   trivially and fails *reasoning* tasks almost completely. The "hard but solvable"
   middle band is nearly empty — which bounds what these agents can be trusted to do.

---

## 2. What was studied, and how

### 2.1 The question
Not "which framework is best" in the abstract, but: *for a bespoke coding agent that
reviews / debugs / tests / optimises / documents real Python data-pipeline code,
driven by a medium local LLM, which design choices actually move success — and by how
much?* The four candidate designs were a hand-rolled loop, LangGraph + deepagents,
Pydantic-AI, and (later, unbuilt) n8n as an orchestration layer.

### 2.2 Design: one variable at a time
A factorial study where each phase freezes everything but one axis:

- **Agent** (Phase 9) — fix model + task + temperature, vary the agent. Froze the
  agent code as `agents-frozen-v1` afterwards so later phases compare against a fixed
  battery.
- **Model** (Phase 10) — run the *identical* frozen battery on a second model.
- **Temperature** (Phase 11) — sweep `MODEL_TEMPERATURE` on the chosen model.
- **Task** (Phase 12) — vary the code/task difficulty.

### 2.3 The harness (how a run is scored)
- **Best-of-N in isolated git worktrees.** Each attempt runs in a fresh detached
  worktree of the target at HEAD (`.venv` symlinked in), so attempts never clobber
  each other or the real tree. Concurrency 2 by default; per-attempt wall-clock cap
  900 s.
- **Automatic check per task.** Either the target's pytest suite passes
  (`check_tests_pass`), a written artefact contains an expected string
  (`check_file_contains`), or — for the Phase-12 tasks — a **frozen behavioural test**
  that the harness restores from HEAD before scoring (`check_frozen_test`), so an agent
  cannot "pass" by deleting or weakening the spec.
- **Per-attempt telemetry.** Outcome, wall-clock seconds, a classified **terminal
  reason** (passed / timeout / recursion-limit / usage-limit / step-limit / no-edit /
  repeat-abort / tests-fail / still-broken), model-turn count, token count, and the
  full diff. This keeps "ran out of budget" from being conflated with "couldn't do it."
- **Statistics.** Success reported as `k/N`; Wilson 95% intervals used to judge whether
  two cells differ. At N=20 the intervals are wide (~±20 pts), so N=20 is a *screen*,
  not a fine measurement.

### 2.4 The models
- **12B** — `gemma-4-12B` (QAT W4A16) on vLLM, L4 24 GB.
- **31B** — `gemma-4-31B` (QAT W4A16) on vLLM, L40S 48 GB.
- Both served OpenAI-compatible, 32k context, with the gemma4 tool-call parser. Runs
  were serialized (`--max-num-seqs 1`) for most of the study; a later change proved
  `--max-num-seqs 4` keeps tool-call arguments intact (24 concurrent multi-tool
  requests verified on the 12B), but scored runs were kept at concurrency 2 for
  comparability.

### 2.5 The tasks
The anchor task across Phases 9–11 was **`debug-best-weather`**: a pipeline's test
suite fails because the test file imports three function names that no longer exist;
the fix is to remap them to the real functions. It is *mechanical* — once you see the
name mismatch, the fix is find-and-replace. Phase 12 added three harder tasks (§6).

---

## 3. The agent battery (`agents-frozen-v1`)

Eight agents across four families, chosen to isolate specific design ideas:

| agent | family | what it is |
|---|---|---|
| `agent_min` | bare loop | hand-rolled chat/tool loop over `httpx` only; 6 tools (read/ls/grep/edit/write/run_tests); a repeated-identical-call guard. The control. |
| `agent_min_improved` | bare loop | `agent_min` + a **finish gate** (don't stop until an edit was made and tests rerun after it) + a repeat **nudge** (corrective feedback instead of a hard abort). |
| `agent_langgraph` | LangGraph | deepagents "deep agent" stack, tuned: trimmed middleware, real shell backend, raw-text read/edit tools, repeat-guard + finish-gate middleware. The framework reference. |
| `agent_langgraph_small` | LangGraph | LangGraph `create_agent` + the same 6 tools, no deepagents planning layer — a 7B-friendly variant. |
| `agent_pydantic` (full) | Pydantic-AI | 6 tools + an **output validator** that refuses to finish until the tests actually pass (verification gate); prose finish. |
| `agent_pydantic_min` | Pydantic-AI | the same tools, no forcing — the closest Pydantic analogue to `agent_min`. |
| `agent_pydantic_typed` | Pydantic-AI | finishes by emitting a **structured `FixReport`** object (`ToolOutput`), no verification. |
| `agent_pydantic_typed_checked` | Pydantic-AI | structured finish **plus** verification (added during the investigation in §5). |

Two cross-cutting ideas under test: **persistence** (finish-gates / nudges that stop an
agent quitting early) and **verification** (refuse to finish until tests really pass).

---

## 4. Results

### 4.1 On a weak model (12B), agent design barely matters
`debug-best-weather`, N=20, T=0.7:

| agent | 12B | Wilson 95% |
|---|---|---|
| `agent_langgraph` | 6/20 (30%) | [15%, 52%] |
| `agent_langgraph_small` | 6/20 (30%) | [15%, 52%] |
| `agent_pydantic` (full) | 6/20 (30%) | [15%, 52%] |
| `agent_pydantic_min` | 4/20 (20%) | [8%, 42%] |
| `agent_min` (control) | 4/20 (20%) | [8%, 42%] |
| `agent_min_improved` | 3/20 (15%) | [5%, 36%] |
| `agent_pydantic_typed` | 0/20 (0%) | [0%, 16%] |

Everything except `typed` sits in a 15–30% band with heavily overlapping intervals —
**not distinguishable at N=20.** The frameworks' apparent edge is within noise. The one
real signal is `typed` at a hard zero. (Slow agents also lost 3–6 attempts each to the
900 s timeout here, so these are lower bounds.) Conclusion at the time: don't tune
further on the weak model; freeze the agents and change the model.

### 4.2 The model is the dominant lever (12B → 31B)
Identical frozen battery, same task, same N, same temperature:

| agent | 12B | 31B | with telemetry (31B: median turns / median ktok) |
|---|---|---|---|
| `agent_langgraph` | 6/20 (30%) | **20/20 (100%)** | 12 turns / 131k |
| `agent_langgraph_small` | 6/20 (30%) | 19/20 (95%) | 17 / 154k |
| `agent_pydantic` (full) | 6/20 (30%) | 18/20 (90%) | 19 / 194k |
| `agent_pydantic_min` | 4/20 (20%) | 16/20 (80%) | 23 / 262k |
| `agent_min_improved` | 3/20 (15%) | 14/20 (70%) | 20 / 183k |
| `agent_min` (control) | 4/20 (20%) | 10/20 (50%) | 17 / 151k |
| `agent_pydantic_typed` | 0/20 | 0/20 | — |

Findings:
- **~3× lift across the board**, and the agents now *separate*: framework agents
  (100/95/90%) vs the bare loop (50%), with non-overlapping intervals
  (langgraph [84,100%] vs min [30,70%]). **Frameworks help — on a capable model.**
- **`min_improved` reverses with model strength.** Persistence *helps* a capable model
  (31B: 14 > min's 10) but *wastes budget* on an incapable one (12B: 3 < min's 4). The
  same mechanism is a feature or a liability depending on whether the model can
  eventually get there.
- **Timeout censoring vanished** on the 31B (0 timeouts vs 3–6 on the 12B): the L40S's
  memory bandwidth offsets the bigger model, so attempts finished well under the wall.
- **Cost of a run** (31B telemetry): ~12–23 model turns and ~130–260k tokens for a
  successful fix. The deepagents stack is *not* the token hog here; the Pydantic
  verification agents (which re-ask until tests pass) use the most (`pydantic_min`
  262k).

### 4.3 The structured-output finish breaks the agent (`typed` deep-dive)
`agent_pydantic_typed` scored a hard **0/20 on both models** — the only
model-independent zero — so it was investigated in depth.

- **Two different failure modes.** On the 12B it *lies*: it emits `FixReport(tests_pass
  =True)` over a still-broken suite (nothing checks the claim). On the 31B it *crashes*:
  forced to answer through the output tool, it emits a storm of empty/non-actionable
  turns; Pydantic-AI counts each against a cumulative output-retry budget and raises.
- **A fixed variant (`typed_checked`)** added verification (deps + an output validator
  that refuses the report until tests really pass) and a clean failure path. It removed
  the lying and the crashing but **stayed at 0/12.** Escalating the budget (output
  retries 8→80, requests 40→200) only traded retry-exhaustion for wall-clock timeout.
  A tool-free structured finish (`PromptedOutput`) fared no better: the model fixates
  on emitting the JSON report and **never calls a tool** (0 edits).
- **Conclusion:** on this model, forcing the final answer into a structured schema —
  by *any* mechanism the framework offers — breaks the tool-use loop. The value was
  never the structured output; it is the **verification gate**, and that has to ride a
  **plain-prose finish** (which is exactly what `agent_pydantic` full does, at 90%).

### 4.4 Temperature: higher is better, greedy is catastrophic
Full battery, 31B, `debug-best-weather`, N=20, `MODEL_TEMPERATURE` ∈ {0.0, 0.3, 0.7, 1.0}:

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

This is the **counterintuitive** result. Success rises *monotonically* with temperature
for every mid-band agent, and **greedy decoding (T=0) is catastrophic** — the bare
loops collapse to 0/20. The effect is large and significant (e.g. `min` 0/20 → 13/20,
intervals [0,16%] vs [43,82%]).

- **Mechanism:** at T=0 the model is deterministic, so the first wrong move — a
  mis-aimed exact-string edit, a repeated tool call — recurs forever and the agent can
  never climb out. Sampling diversity is the *only* escape from a stuck state.
- **Scaffolding substitutes for temperature.** `langgraph`/`_small` are flat at ~100%
  across the whole range — their planning + retry structure recovers from stuck states
  *structurally*, so the knob is irrelevant. The weaker the scaffold, the more
  temperature carries it.
- **Structural zeros don't move.** `typed`/`typed_checked` are 0 at every temperature —
  confirming their defect is structural, not a sampling one.

### 4.5 The capability ceiling (Phase 12: three harder tasks)
To find a task in the "hard but solvable" band (so a temperature *optimum* might
appear), three tasks were designed (workflow-vetted), each on its own labelled branch
in the target repo with a frozen, tamper-proof spec, and each verified solvable by a
reference fix before the agents saw it:

| task | kind | result | failure character |
|---|---|---|---|
| **#1 inline-vs-S3 threshold** (multi-file feature) | hard | **0/96** frozen budgets, **0/96** relaxed | finishes but produces wrong fixes; can't coordinate publish+consume |
| **#2 S3 retry/backoff** (single-file, known pattern) | easy | **~100%** (all 3 agents 2/2) | trivial |
| **#3 message_id trace bug** (subtle single-file) | hard | **0/24** (default + relaxed + guided prompt) | can't localize; even pointed at the spot, can't fix |

- **#1 is capability-, not budget-walled.** The first sweep failed on step/recursion/
  usage limits; relaxing every agent budget 3× flipped the failures to "finished but
  wrong" and the score *stayed 0 at every temperature.* More budget ≠ more capability.
- **#3 is subtle twice over.** At default/relaxed budgets the agents mostly *don't edit*
  (can't find the bug — a `Message()` constructor silently regenerating `message_id`).
  Pointing the prompt straight at the consume-step rebuild made them edit and run
  tests, but they still couldn't produce the correct fix.
- **The band is nearly empty for this model.** Autonomous debugging here is **bimodal**:
  mechanical / pattern tasks are trivial (retry; the stale-import remap), reasoning
  tasks are effectively impossible (multi-file coordination; localizing a subtle
  semantic bug; fixing one even when handed its location). There is little middle
  ground — which is *why* only the mechanical anchor task produced a temperature
  gradient: there was no constructed task between "trivial" and "impossible" on which
  to look for a temperature optimum.

---

## 5. Synthesis — design rules

1. **Spend on the model first.** It dwarfs every other lever measured. A capable model
   turns a coin-flip into a near-certainty and is what makes a framework worth its
   weight.
2. **Frameworks earn their keep on a capable model, not a weak one.** On the 31B,
   LangGraph/deepagents and the verification-gated Pydantic agent clearly beat the bare
   loop; on the 12B they were indistinguishable from it. Don't adopt the dependency
   weight until the model can use it.
3. **Verification > structured output.** Refusing to finish until the tests actually
   pass is the single most useful agent-side trick. Delivering it as a structured
   "report" object is actively harmful — keep the finish in prose and gate it with a
   validator.
4. **Run hot, not cold.** On a medium local model, low temperature removes the agent's
   only escape from a loop. Pair weak scaffolding with high temperature (~1.0), or use
   scaffolding that recovers structurally and stop worrying about the knob.
5. **Persistence cuts both ways.** Finish-gates and nudges help a model that can
   eventually get there and waste budget on one that can't — tune them to the model.
6. **Know the capability boundary.** This model is reliable for *mechanical* code fixes
   and unreliable for *reasoning* ones. Scope autonomous use to the former; keep a human
   in the loop for the latter.

---

## 6. Practical recommendation (current state)
For this use case and this model class, the best-performing design is **a framework
agent with a verification gate, run at high temperature**: `agent_langgraph`
(deepagents, structurally robust, temperature-insensitive, 100% on the anchor task) or
`agent_pydantic` full (prose finish + must-pass validator, 90%, simpler/lighter). The
hand-rolled loop is a fine *control* and a useful minimal fallback, but leaves ~40–50
points of success on the table versus the frameworks on a capable model. Avoid any
structured-output finish.

---

## 7. Limitations & threats to validity
- **One task type carried most of the study.** Phases 9–11 all used the single
  mechanical `debug-best-weather` task; the agent ranking and the temperature curve may
  not generalise to other task kinds (the Phase-12 ceiling result is a warning that
  they may not).
- **N=20 is a screen.** Wilson intervals are ~±20 points; only large effects (the model
  lift, the temperature collapse) are statistically solid. Mid-band agent differences
  are suggestive, not proven.
- **Two models, one family.** Only gemma-4 12B/31B were compared; the "model is the
  lever" result is a within-family scaling result, not a cross-vendor one.
- **Serving was serialized** for scored runs; throughput, not capability, was the
  constraint on N.
- **n8n** (the orchestration layer) was never built, so the four-way comparison is
  really three-way.
- **Phase-12 task difficulty was hand-calibrated** and the band proved hard to hit; the
  "band is empty" claim is strong but rests on three constructed tasks, not a
  systematic difficulty sweep.

---

## 8. Open questions / next
- **A genuinely middle task.** The band is so narrow that hitting it probably needs a
  *mechanical* bug with a *non-obvious location* (search-hard, fix-easy) rather than a
  reasoning-hard fix — one more design pass on task construction.
- **A second model family** (a Llama or Mistral coder of similar size) to test whether
  the bimodal capability and the temperature result are gemma-specific or general.
- **Other task kinds at N≥20** (test-writing, optimise, document/review) to see whether
  the agent ranking holds outside mechanical debugging.
- **Concurrency 4** for ~2× throughput once the argument-integrity check passes on the
  31B (proven on the 12B).
- **n8n** orchestration layer, if the agent brain is judged good enough to wrap.

---

## 9. Appendix — provenance
Numbers are from the per-attempt JSONL artifacts: `RESULTS.md.jsonl` (12B battery),
`RESULTS-31b.md.jsonl` (31B battery + Phase-11 T=0.7 column),
`RESULTS-temp-{0.0,0.3,1.0}.md.jsonl` (Phase 11), `RESULTS-task1-temp-*` /
`RESULTS-task1relax-temp-*` (Phase 12 #1), and the `RESULTS-task2-smoke` /
`RESULTS-task3-*` smokes (Phase 12 #2/#3). Agent code is frozen at `agents-frozen-v1`;
the Phase-12 tasks live on the `harness-task-{inline,s3retry,msgid}` branches of the
target repos with their frozen specs.
