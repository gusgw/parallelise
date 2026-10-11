# RESULTS — agent harness experiments

Measured findings from building and running the harnesses. Orientation and how-to
are in `README.md` at the top of this study folder.

**Setup:** the harnesses ran on a development machine; the model was
**Mistral-7B-Instruct-v0.3** (Q4, llama.cpp `--jinja`) served at
`http://localhost:4000/v1`. All "live" runs below use a throwaway repo containing
`note.txt` = `"the secret code is 4242"` (the real target
codebases were unavailable during these runs, so real-target tasks were still
pending, see §5).

---

## 1. Headline: what runs on a medium (7B) local model

| Harness config | Prompt + tool surface | Tool call emitted? | End-to-end on Mistral-7B |
|---|---|---|---|
| `agent_min` (raw httpx loop, 6 tools, short prompt) | small | **yes** | ✅ "secret code is 4242" |
| `agent_langgraph` · `build_smalltools_agent` (LangGraph `create_agent` + 3 own tools) | small | **yes** | ✅ "secret code is 4242" |
| `agent_pydantic` · `build_agent` (Pydantic AI + 3 tools) | small | **yes** | ✅ "secret code is 4242" |
| `agent_langgraph` · `build_minimal_agent` (LangChain + deepagents `FilesystemMiddleware`) | medium | no (narrates) | ❌ |
| `agent_langgraph` · `build_agent` (deepagents **default** stack) | large | no (narrates) | ❌ |

**Conclusion:** on a 7B the *framework doesn't matter* — **prompt size and tool
surface do.** A small surface works through a raw loop, LangGraph, *and* Pydantic
AI. The deepagents default stack (and even its filesystem middleware) inject large
prompts that push the 7B into "narration" (it describes calling a tool instead of
emitting a tool call). Those need a stronger model (Gemma 4 / coder) — see
section 5.

---

## 2. Model-level findings (`common/MODEL_NOTES.md`)

1. **Tool-calling mechanics work** on this endpoint — the model emits well-formed
   OpenAI `tool_calls` and completes a full round trip. (Validated by
   `common/probe_model.py` and `model.supports_tool_calls()`.)
2. **It depends on server-side `--jinja`** (renders Mistral's
   `[AVAILABLE_TOOLS]`/`[TOOL_CALLS]` protocol). Currently on. Without it the model
   *narrates* and nothing runs (the cause is the missing chat template).
   Preflight with `model.supports_tool_calls()` before real work.
3. **Tool-use decisioning is prompt-sensitive.** Vague tasks → hallucination;
   a **long/forceful** "you MUST use tools" prompt **backfires** (narration);
   a **short** prompt + explicit task works. Keep prompts short.
4. **Tool-call ids must be 9-char alphanumeric** (the Mistral template 400s
   otherwise). The server returns short ids (`"1"`), so any multi-turn tool use
   breaks unless the client rewrites them. Fixed framework-agnostically by
   `harness_common.mistral_compat` (an httpx transport that relabels ids in
   outgoing `/chat/completions` bodies; injected into LangChain `ChatOpenAI` and
   Pydantic AI `OpenAIProvider` via `http_client`).
5. **`tool_choice:"required"` times out** on this server; forcing a *named*
   function works. The endpoint used here was `http://localhost:4000/v1`.

---

## 3. Per-harness status

- **`common`** (httpx-only, sync, framework-free): `model` client + tool-calling
  preflight + id-relabel, `tools/{proc,fs,run_tests,shell}`, `targets`, `asyncapi`
  (parse/render pure; `generate` shells to `faststream docs gen`), `mistral_compat`
  id-fix transport. **40 tests** pass.
- **`agent_min`** (baseline, no framework): hand-rolled chat→tool→chat loop, 6
  tools. **9 tests**; live ✅. The fallback that needs almost no dependencies.
- **`agent_langgraph`** (reference): deepagents 0.6.8 / langgraph 1.2.4 /
  langchain-openai 1.2.2. Three build functions (`build_agent` full deepagents,
  `build_minimal_agent` FS-middleware-only, `build_smalltools_agent` minimal).
  **4 tests**; `build_smalltools_agent` live ✅, deepagents stack ❌ (narrates).
- **`agent_pydantic`** (challenger): Pydantic AI 1.106.0, 3 tools. **4 tests**
  (incl. a `TestModel` tool run); live ✅.

Total: **57 tests** across the four projects; every `check.sh` green.

---

## 4. Dependency sets resolved (what must be installed)

- `common` / `agent_min`: **`httpx` only** (+ `pytest` for dev).
- `agent_langgraph`: **deepagents 0.6.8, langgraph 1.2.4, langchain-openai 1.2.2,
  pydantic 2.13.4, openai 2.41.0** + a large transitive tree (tiktoken, orjson,
  ormsgpack, tenacity, uuid-utils, …).
- `agent_pydantic`: **pydantic-ai 1.106.0** + its tree (openai, httpx, …).

The baseline needs almost nothing; the frameworks need a substantial set of packages.

---

## 5. Not yet done / pending

- **Real-target tasks** (the `best_weather` failing-test fix, `mealworm` review,
  AsyncAPI generation): need the target codebases to be present and their own
  dependencies installed (for example after `uv sync` in the target).
- **Gemma 4 endpoint** (requested at the time): once provided, flip
  `MODEL_BASE_URL`/`MODEL_NAME` and re-run everything — the deepagents reference
  should then drive its full stack.
- **The eval harness (Phase 4) is built and unit-tested** (`eval/`, 9 tests) — task
  set, a resilient runner, checks, and markdown output; the actual scoring run is
  pending the target codebases and the model.
- **n8n orchestration** (Phase 6) is not yet built.
