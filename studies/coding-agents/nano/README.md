# nano — the one-file starter agent

The smallest useful coding agent: an `httpx` tool-calling loop in a single file,
with nothing else in this repo required. It is the quickest way to get
something running.

## Files
- `nano_agent.py` — model call → three tools (read / write / run_tests) → loop.
- `test_nano.py` — fast, offline tests (the loop runs against a fake model).
- `check.sh` — syntax + import + tests.

## Run it
```
MODEL_BASE_URL=http://localhost:4000/v1 MODEL_NAME=your-model \
    python nano_agent.py /path/to/repo "make the tests pass"
```
Reads `MODEL_BASE_URL`, `MODEL_NAME`, `MODEL_API_KEY` (optional), and
`MODEL_TEMPERATURE` (default 0.7) from the environment.

## Check it
```
bash check.sh
```
Only needs `httpx` and `pytest`; no endpoint is contacted.

## When you outgrow it
`nano` keeps everything in one file and uses whole-file writes. For exact-snippet
edits, more tools, repeat/finish guards, and reusable pieces, move up to `common/`
+ `agent_min/` .
