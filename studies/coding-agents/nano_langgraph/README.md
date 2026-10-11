# nano_langgraph — minimal LangGraph agents (two flavours)

Bare-bones LangGraph coding agents in one file, over the same model:

- **`build_deep()`** — `deepagents.create_deep_agent`. Filesystem tools and a shell
  `execute` are built in, so you give it only a model and a prompt. The smallest form
  of the "deepagents" reference.
- **`build_plain()`** — `langchain.agents.create_agent` + three plain tools
  (read / write / run_tests). No deepagents; lighter to install.

## Files
- `lg_agent.py` — shared model + both flavours + the runner.
- `test_nano_langgraph.py` — offline tests (both flavours construct cleanly).
- `check.sh` — syntax + import + tests.
- `requirements.txt` — pinned `langgraph` / `langchain` / `deepagents`.

## Set up & run
```
pip install -r requirements.txt
MODEL_BASE_URL=http://localhost:4000/v1 MODEL_NAME=your-model \
    python lg_agent.py plain /path/to/repo "make the tests pass"   # or: deep
```
Reads `MODEL_BASE_URL`, `MODEL_NAME`, `MODEL_API_KEY` (optional), `MODEL_TEMPERATURE`.

## Check it
```
bash check.sh          # needs the requirements + pytest; no endpoint contacted
```

Start with `plain` (fewer dependencies). Use `deep` to see the deepagents stack — it
brings the larger install but built-in tools and a shell.
