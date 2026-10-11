# nano_pydantic — minimal Pydantic-AI agent (with the verify gate)

A bare-bones Pydantic-AI coding agent in one file. Same three tools as `nano/`
(read / write / run_tests), but driven through Pydantic-AI, plus the pattern that
made Pydantic effective in testing: an **output validator that refuses to finish
until the tests pass**.

## Files
- `pyd_agent.py` — model → state → agent (tools + the verify-gate validator).
- `test_nano_pydantic.py` — offline tests using Pydantic-AI's `TestModel`.
- `check.sh` — syntax + import + tests.
- `requirements.txt` — pinned `pydantic-ai`.

## Set up & run
```
pip install -r requirements.txt
MODEL_BASE_URL=http://localhost:4000/v1 MODEL_NAME=your-model \
    python pyd_agent.py /path/to/repo "make the tests pass"
```
Reads `MODEL_BASE_URL`, `MODEL_NAME`, `MODEL_API_KEY` (optional), `MODEL_TEMPERATURE`.

## Check it
```
bash check.sh          # needs pydantic-ai + pytest; no endpoint contacted
```

The verify gate is the point: `run_tests` records whether the suite passed, and the
validator raises `ModelRetry` until it has — so the agent cannot stop while the tests
are red.
