# MODEL_NOTES — local endpoint behaviour (Phase 1.2 gate)

Recorded 2026-06-08. Endpoint probed live from the development machine.

## Endpoint
- URL: `http://localhost:4000/v1` (OpenAI-compatible)
- Served id: **`mistral`** (use this as `MODEL_NAME`, *not* `mistral-7b-instruct-v0.3`)
- Server: **llama.cpp** (`owned_by: llamacpp`), GGUF, Q4 (~4.37 GB), ~7.25B params
- Context: **32768** tokens (`n_ctx`)
- `/v1/models` advertises `capabilities: ["completion"]` only — but tool calling
  does work (below), so don't trust that field.

## Go/no-go result: **GO** (tool calling works)
- **Plain chat:** OK.
- **Tool calling:** the model emits well-formed OpenAI `tool_calls` — e.g. for
  "what is 21 + 21? use the add tool" it returns
  `tool_calls[0] = {function: {name: "add", arguments: '{"a": 21, "b": 21 }'}}`.
- **Full round trip** (assistant tool_calls → tool result → final answer): OK,
  *after* the id workaround below. Final reply: "The sum of 21 and 21 is 42."

## Critical quirk: tool-call IDs must be 9-char alphanumeric
- The endpoint **returns** short ids (e.g. `"1"`).
- But the **Mistral chat template validates ids on the way back** and raises:
  `Tool call IDs should be alphanumeric strings with length 9!` → HTTP 400.
- This fails regardless of how the assistant/tool follow-up messages are otherwise
  shaped (content `None` vs `""`, tool keyed by `tool_call_id` vs `name`).
- **Workaround (baked into `harness_common.model`):** `relabel_tool_calls()`
  rewrites each tool_call `id` with `new_tool_call_id()` (`uuid4().hex[:9]`) before
  echoing the assistant message, and the matching `tool` message uses the same new
  id. The agent loop (Phase 0) must send tool results back this way.
- This is Mistral-template-specific; revisit when we move to Gemma 4.

## Server dependency: tool-calling needs `--jinja`
Tool-calling only works because the `llama-server` process runs with **`--jinja`** (it
renders Mistral v0.3's `[AVAILABLE_TOOLS] … [/AVAILABLE_TOOLS]` / `[TOOL_CALLS]…`
protocol and parses replies back into `tool_calls`). This is **server-side, outside
our code**. Verified live 2026-06-08: the endpoint returns real `tool_calls` and
the embedded template's id-validation fires — so `--jinja` is currently ON (it was
already on at the endpoint). If the server is restarted
without it, the symptom is the **narration** failure mode (prose "I'll call
read_file …", no `tool_calls`) and nothing runs. Detect it before real work with
`model.supports_tool_calls()` (or `python probe_model.py`). The 9-char-id rule
comes from that same embedded template; if they switch to `--chat-template-file`
or swap models, re-check the id rule (relabel stays safe).

## Endpoint
- **`http://localhost:4000/v1`** — set through `MODEL_BASE_URL`. When the agent runs
  inside a container, use an address that the container can reach.

## Misc
- `temperature: 0` is accepted and used for deterministic probing.
- No API key required (`MODEL_API_KEY` empty).

## Tool-use decisioning (agent loop) — 2026-06-08
Mechanics work, but *deciding* to use a tool on this 7B is prompt-sensitive:
- Vague task → no tool call; the model hallucinates an answer or tells the user to
  run a shell command.
- Explicit task naming the tool ("Use the read_file tool …") → calls it. GOOD.
- **Long/forceful system prompt backfires:** the model narrates "I have used the
  read_file tool …" WITHOUT emitting a call. Keep system prompts SHORT (this is
  why `agent_min`'s `DEFAULT_SYSTEM` is short).
- `tool_choice: "required"` (force *any* tool) → **request times out** on this
  llama.cpp build — do not use.
- `tool_choice: {type: function, function: {name: ...}}` (force a *specific* tool)
  → works and is fast — a reliable lever when you know which tool to force.
- With a short prompt + explicit task, the full loop returns the correct answer
  (validated via `agent_min` on a throwaway repo).

Implication: Mistral-7B is usable for guided, single-tool steps but unreliable at
autonomous multi-tool decisioning — strong motivation for Gemma 4 12B / a
coder-tuned model (Phase 7) for the real agent.

## Still to probe
- Multi-tool *chains* (does it pick the right tool among several on a real task?).
- Structured output (`response_format`) behaviour.
- The full tool surface on a real target (needs the target's deps installed).
