# agent_langgraph — run notes

## Status
Reference deep agent =
`deepagents.create_deep_agent(model=ChatOpenAI(Mistral), backend=LocalShellBackend(repo, virtual_mode=True))`.
Construction tests green (`bash check.sh`, 3 tests).

## Version set installed (the packages needed)
deepagents **0.6.8**, langgraph **1.2.4**, langchain-openai **1.2.2**, langchain-core,
pydantic **2.13.4**, openai **2.41.0**, tiktoken, orjson, ormsgpack, tenacity,
uuid-utils, wcmatch, zstandard, … (full transitive tree resolved by uv —
substantial).

## KEY FINDING (2026-06-08): deepagents does NOT work on Mistral-7B
Live smoke on a throwaway repo, task *"Read note.txt and report the secret code"*:
- The model **narrates** — replies in prose "I will use the `read_file` tool …
  ```read_file(file_path=…)```" and emits **no tool call** (messages: Human, AI
  only; no ToolMessage). Nothing runs.
- Cause: deepagents prepends a large opinionated system prompt + ~8 tools
  (planning/`write_todos`, ls/read/write/edit/glob/grep, `execute`, subagent
  `task`). On a 7B this triggers the same narration/backfire seen with long
  prompts in `agent_min` (see `common/MODEL_NOTES.md`). `--jinja` IS on
  (agent_min calls tools fine on this endpoint), so this is a prompt-size /
  tool-surface decisioning failure, **not** a server issue.

## Headline comparison
- `agent_min` (tiny prompt, 6 tools): **works** on Mistral-7B (returns 4242).
- `deepagents` (large prompt, many tools + planning): Mistral-7B **narrates → fails**.
Concrete evidence for the research's warning that deepagents is tuned for large
models. The reference likely needs **Gemma 4 / a coder model** (Phase 7) or a
**trimmed middleware stack**.

## Trimming experiment (1) — results (2026-06-08)
- `build_minimal_agent` (LangChain `create_agent` + deepagents `FilesystemMiddleware`
  only): **still narrates** on the 7B — `FilesystemMiddleware` injects a verbose
  tool-usage prompt.
- `build_smalltools_agent` (`create_agent` + our own 3 tiny tools + short prompt,
  no middleware): the 7B **emits a tool call** (narration solved) — but then hit the
  Mistral **9-char tool-call-id 400** on the turn that sends the tool result back (LangChain echoes the
  server's short id).
- **Fix:** `harness_common.mistral_compat` — an httpx transport that relabels
  tool-call ids to 9-char in outgoing `/chat/completions` bodies, injected into
  `ChatOpenAI` (`http_client`/`http_async_client`). With it,
  `build_smalltools_agent` completes **end-to-end on the 7B**:
  Human → AI(tool_call) → Tool → AI "The secret code is 4242."
- **Conclusion:** LangGraph is usable on Mistral-7B with a SMALL prompt + tool
  surface. The deepagents *default* stack and even `FilesystemMiddleware` are too
  verbose for the 7B (narration). The full deepagents reference still needs
  Gemma 4 / a coder model; `build_smalltools_agent` is the 7B-friendly variant.

## How to run (needs target deps + reachable model)
    bash check.sh
    uv run --extra dev python -m agent_langgraph --target best_weather \
        "Find and fix the failing test."

## Temperature (Phase 8, 2026-06-10)
- `build_model()` sets `ChatOpenAI(temperature=hc_model.default_temperature())` (env
  `MODEL_TEMPERATURE`, default 0.7) instead of a hardcoded 0; runs are bounded by
  `--recursion-limit`. `SHORT_SYSTEM` now ends with the shared `prompts.STRATEGY`.
  Rationale in the temperature section of the study README.

## Gemma 4 12B tuning (Phase 8 investigation, 2026-06-12)
Conversation-log-driven (see top-level FINDINGS.md for the full catalogue). The full
deepagents agent went 0/3 -> 3/10 on the best_weather debug task; the smalltools
variant matches at 3/10 — configuration dominated framework choice.
- **Profile trim**: `_register_small_model_profile` registers a deepagents
  `HarnessProfile` keyed `openai:<model>` dropping `TodoListMiddleware` (one
  iteration cost 3 recursion steps; ~16 real turns at limit 50), the dead
  summarization middleware (170k trigger vs 32k context), and the `task`
  sub-agent tool. After the trim: 2 steps/iteration, ~25 turns.
- **execute env**: `LocalShellBackend(..., inherit_env=True)` — the default
  empty env made bare pytest resolve to the SYSTEM pytest. The system prompt
  names the exact command (from the target's `test_cmd`).
- **Raw-text tool swap**: middleware `read_file` is cat -n formatted while
  `edit_file` wants raw text; the 12B copied phantom indentation and looped.
  `excluded_tools={"read_file","edit_file"}` + our `_override_tools(root)`
  (raw read, forgiving edit that accepts a `content` alias for `new_string`,
  '/'-style paths normalized).
- **Guards** (`agent_langgraph.middleware`): `RepeatGuardMiddleware` intercepts
  an exact-repeat tool call with corrective feedback; `FinishGateMiddleware`
  refuses a prose finish until an edit happened and tests reran after it
  (the langchain-side equivalent of a Pydantic output validator).
- **Caps**: `max_tokens=3000` (one untuned run ended in a 46k-char essay).
- **Prompt**: `DEEPAGENTS_SYSTEM` agrees with the middleware ('/'-rooted paths),
  has no backticks (the model greps for them literally), and names the test
  command. The shared relative-path STRATEGY is wrong for this stack — do not
  reuse it here.
