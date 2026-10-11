# agent_pydantic — run notes

> **About this file.** This is a set of run notes, kept as a working record and not edited into
> finished prose. It uses abbreviations and a terse note style. The readable account of the
> study is [../README.md](../README.md). Abbreviations: N is the number of attempts per cell; CI is a 95% confidence interval; QAT is
> quantisation-aware training; W4A16 means 4-bit weights with 16-bit activations; T is the
> sampling temperature; B means billion parameters.

## Status
Challenger = **Pydantic AI 1.106.0**, small tool surface (`read_file` / `ls` /
`grep`) over `harness_common`; model via `OpenAIChatModel` + `OpenAIProvider(base_url=…,
http_client=mistral_compat id-fix)`. Tests green (`bash check.sh`, 4: construction +
a `TestModel`-driven tool run, no network).

## Live on Mistral-7B (2026-06-08)
Task *"Use the read_file tool to read note.txt and report the secret code"* →
**"The secret code is 4242."** End-to-end success. The Mistral 9-char
tool-call-id quirk is handled by the shared `harness_common.mistral_compat` httpx
client, injected via `OpenAIProvider(http_client=…)`.

## Notes
- Same recipe that works for `agent_min` and the LangGraph small-tools agent:
  short instructions + 3 tools. Pydantic AI's `TestModel` gives clean offline tool
  tests (`tests/test_agent.py`).
- The Pydantic deep-agent layer (`pydantic-ai-harness` / Vstorm `pydantic-deep`)
  is **not** used yet — this is the minimal challenger. A larger surface will
  likely need Gemma 4, exactly like deepagents.

## How to run (needs target deps + reachable model)
    uv run --extra dev python -m agent_pydantic --target best_weather \
        "Use the tools to find and fix the failing test."

## Temperature (Phase 8, 2026-06-10)
- `build_model()` sets `OpenAIChatModel(settings=ModelSettings(
  temperature=hc_model.default_temperature()))` (env `MODEL_TEMPERATURE`, default
  0.7) — previously it used the provider default, so the three agents now match.
  `__main__` bounds runs with `UsageLimits(request_limit=25)`. `SHORT_SYSTEM` ends
  with the shared `prompts.STRATEGY`. Rationale in the temperature section of the study README.

## Gemma 4 12B tuning (Phase 8 investigation, 2026-06-12)
The original agent was READ-ONLY (read/ls/grep) while its prompt claimed it
could edit and test — its 0/3 was decided before the model spoke. Now it has
agent_min's six-tool surface plus fix-then-verify forcing:
- **Framework retry channel**: `_safe` raises `ModelRetry` instead of returning
  "ERROR:" strings (which look like success and bypass retries entirely).
- **Verify gate**: `RunState` deps track whether run_tests ran since the last
  edit; an `@agent.output_validator` rejects finishing unverified — closing the
  default loophole where ANY prose response ends the run as success.
- **Forcing**: callable `model_settings` keep `tool_choice="required"` until
  verified (static "required" raises UserError; the callable form is the
  sanctioned escape hatch). `parallel_tool_calls=False`, `max_tokens=3000`.
- **Guards**: exact-repeat calls and oversized edit snippets (>4000 chars)
  raise `ModelRetry` — one untuned run blew the 32k context by passing a whole
  file as edit args (history is re-sent in full every request; 1.106 does no
  trimming).
- Eval (N=10): 0/10 BUT failures changed shape — it now edits and runs tests
  (rc=1 fast fails, median 178s) instead of narrating for 13 minutes. Suspect
  the 25-request budget is too tight with required-tool forcing; one conversation log
  pass needed. See FINDINGS.md.
