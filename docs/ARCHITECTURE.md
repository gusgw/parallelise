# Architecture and build contract

This document is the specialist's map of the code: the layout of the repository and the
interfaces between its parts. Every path below is relative to the repository root.

## What the system does

A problem is a research Jupyter notebook that implements a numerical algorithm in
serial Python and, in its final cell, defines a plain `ORACLE` dictionary holding its
parameters, its own computed result and a tolerance. A backend is one of `openmp`,
`mpi` or `cuda`. An agent, driven by a language model reached through an
OpenAI-compatible endpoint, is given a work folder containing the notebook, the oracle
in C form, a kernel header with the exact signature to implement, a benchmark it cannot
change, and a Makefile. The agent writes the kernel and a unit test and must make
`make check` pass. The gate then judges the work independently: it builds with warnings
treated as errors, runs the unit test at one worker and at several, runs the benchmark,
and validates the speed-up figures. Every run is recorded as one structured record.

## Layout

```
README.md                 what it does, results for a general reader, how to run it
LICENSE                   MIT
pyproject.toml            package metadata, dependencies, pytest and ruff settings
Makefile                  convenience targets: install, lint, test, check-references, image
Dockerfile                CPU stage (gcc, OpenMPI, Python) and an optional CUDA stage
k8s/job.yaml              run one conversion cell as a Kubernetes Job
.github/workflows/ci.yml  lint, tests, reference gates, container build
docs/ARCHITECTURE.md      this file
docs/DESIGN.md            the experimental design and what was learned
docs/RESULTS.md           every results table, with provenance
docs/OPERATIONS.md        deploying, health checks, reading the logs, troubleshooting
src/parallelise/          the package (see "Modules")
src/parallelise/data/     files copied into every work folder: testing.h, bench_sweep.sh
problems/<name>/          one folder per algorithm (see "Problems")
results/                  the run records and the per-run evidence (see "Results")
studies/coding-agents/    the earlier study of agent designs on a debugging task
bindings/                 nanobind wrappers that call the finished kernels from Python
tests/                    pytest suite; `pytest -q` from the root runs it
```

## Modules

All code is in `src/parallelise/`. Python 3.11 or later. Every public function has
type hints and a docstring in plain English. Logging goes through
`logging.getLogger("parallelise.<module>")`; nothing prints except the command-line
layer. Subprocesses are run with argument lists, never a shell string, and always with
a timeout. No module reads the network except `model.py`.

- `config.py` — `Settings` dataclass and `Settings.from_env(overrides)`. Fields:
  `gateway_url` (default `http://localhost:4000/v1`), `model` (default `local`),
  `temperature` (default 0.2), `api_key` (default empty), `request_timeout_s`
  (default 600), `work_root` (default `work`), `results_path` (default
  `results/runs.jsonl`). Environment variables: `PARALLELISE_GATEWAY_URL`,
  `PARALLELISE_MODEL`, `PARALLELISE_TEMPERATURE`, `PARALLELISE_API_KEY`,
  `PARALLELISE_TIMEOUT`, `PARALLELISE_WORK_ROOT`, `PARALLELISE_RESULTS`. The older
  names `MODEL_BASE_URL`, `MODEL_NAME`, `MODEL_TEMPERATURE` and `MODEL_API_KEY` are
  accepted as fallbacks. Validation raises `ConfigError` with a plain message: the URL
  must start with http or https, the temperature must lie in [0, 2], the timeout must
  be positive.
- `log.py` — `configure_logging(level: str, json_lines: bool = False)`.
- `model.py` — `ModelClient(settings)` with `chat(messages, tools) -> ChatResult`
  and `health_check() -> HealthReport`. `ChatResult` holds `message` (the assistant
  message dict), `prompt_tokens`, `completion_tokens`, `latency_s`. The client keeps a
  list of `CallRecord(latency_s, prompt_tokens, completion_tokens, status)` and a
  `summary()` returning requests, total tokens, mean and median latency, and
  completion tokens per second of generation time. `health_check` calls `/models`,
  confirms the configured model is listed, makes one tiny chat call, and reports
  reachability, the model list, and the round-trip time. Retries: three attempts with
  back-off on connection errors and HTTP 5xx; none on 4xx.
- `results.py` — `RunRecord` dataclass and `ResultsStore(path)` with `append(record)`
  and `load() -> list[RunRecord]`, one JSON object per line. Fields of `RunRecord`:
  `when` (ISO 8601), `problem`, `backend`, `model`, `agent`, `agent_version`
  (short hash of the agent source), `masked` (bool), `result` ("PASS" or "FAIL"),
  `failure_stage` (one of `none`, `no-files`, `tampered`, `build`, `test-one-worker`,
  `test-many-workers`, `benchmark`, `timeout`, `agent-error`), `speedup_best`
  (float or null), `wall_time_s`, `requests`, `prompt_tokens`, `completion_tokens`,
  `latency_mean_s`, `tokens_per_second`, `serving` (provider identifier or
  "not recorded"), `quantisation` (string or "not disclosed"), `notes`.
- `oracle.py` — `extract_oracle(notebook: Path) -> Oracle` executes the notebook's
  code cells in a fresh namespace with a timeout and reads `ORACLE`; validates that
  every expected value is finite and that, where `terms` are given, their sum
  reproduces the expected value within tolerance; `write_reference(oracle, dest)`
  writes `reference.json` and `reference.h`. It rejects an oracle whose values are not finite or whose terms do not sum to the
  expected value.
- `assemble.py` — `assemble_work_dir(problem_dir, backend, dest, include_reference=False)`
  builds a flat work folder: package data files, the notebook, `reference.*`,
  `problem.h` if present, the backend's header, `bench.c` and `Makefile`; with
  `include_reference=True` it also copies the known-good solution, which is how the
  machinery is tested without a model.
- `gate.py` — `run_gate(work_dir, backend, timeout_s=300, speedup_floor=None) ->
  GateResult(passed, failure_stage, speedup_best, log)`. Before anything else it checks the
  trusted files (the benchmark, headers and Makefile) against the digests written when the
  work folder was assembled, and fails at the stage `tampered` if any has changed. It then
  builds with the backend's
  compiler, runs the unit test at one worker and at several, runs the benchmark and
  calls `bench_check`. The floor comes from the problem's Makefile (`SPEEDUP_FLOOR`), and falls back to 1.5 for
  OpenMP and MPI or 3 for CUDA.
- `bench_check.py` — `check_bench_csv(path, floor) -> BenchVerdict(ok, reasons,
  speedup_best)` applies these rules: rejects a reported speed-up that disagrees with
  the measured times, duplicated widths, every parallel time exactly the serial time
  divided by the width, constant throughput, efficiency not equal to one at one worker,
  and best speed-up under the floor. `check_bench_cuda` keeps the CUDA variant.
- `mask.py` — `mask_problem(problem_dir, dest, alias) -> MaskReport`. Produces a copy
  of a problem whose kernel, files, header guards, Makefile variables and benchmark
  refer to the neutral name `alias`; whose notebook markdown is replaced by one neutral
  sentence; whose code comments are removed; and whose identifiers that name the
  algorithm are renamed from a word list kept in the module. Numbers, logic and the
  oracle values are unchanged, and the report lists every substitution made. A masked
  problem must still pass its reference gate.
- `agents/base.py` — `Agent` protocol: `run(work_dir: Path, task: str, settings:
  Settings, client: ModelClient, step_budget: int) -> AgentResult(finished,
  steps, notes)`. `agents/__init__.py` exposes `get_agent(name)` for `basic`,
  `pydantic` and `langgraph`; the two framework agents import their frameworks lazily
  and raise `AgentUnavailable` with an installation hint if the extra is missing.
  All three share one tool surface: read the notebook, read a file, write a file,
  run an allow-listed build command. All three re-run the checks when the model says it
  has finished and feed failures back, but they stop differently. `basic` stops when the
  checks pass or after `step_budget` model calls (default 80). `langgraph` makes up to
  `MAX_PASSES` (8) passes, each limited to `step_budget` steps. `pydantic` stops when the
  checks pass or when the framework's request limit, set to `step_budget`, is reached.
  `docs/DESIGN.md` explains the differences.
- `report.py` — `build_tables(records) -> str` (Markdown: pass counts by model and
  backend, by agent design, by problem) and `build_chart(records, dest_svg)` (one
  grouped bar chart of pass rate by model and backend, drawn with matplotlib).
- `cli.py` — `main(argv=None)` using argparse with subcommands: `health`, `oracle
  <notebook>`, `assemble <problem> <backend> <dest>`, `run <problem> <backend>
  --agent --model`, `grade <work_dir> --backend`, `mask <problem> <alias> <dest>`,
  `bench-check <csv>`, `report`. Exit status 0 on success, 1 on a failed gate or a
  reported error, 2 on bad arguments. Every error message says what to do next.

## Problems

```
problems/<name>/<name>.ipynb      the notebook; its last cell defines ORACLE
problems/<name>/reference.json    generated by `parallelise oracle`; kept for reference
problems/<name>/reference.h       generated likewise
problems/<name>/problem.h         optional input helpers shared by all backends
problems/<name>/<backend>/<name>.h     the kernel signature the agent must implement
problems/<name>/<backend>/bench.c      the trusted benchmark (the agent never edits it)
problems/<name>/<backend>/Makefile     build, check (one worker and several), benchmark
problems/<name>/<backend>/reference/   known-good <name>.c and test_<name>.c
```

The nine problems are integrate, matmul, jacobi, nbody, cg, scan, histogram, decay and
reservoir. The decay header must not describe the parallelisation technique; it states
the recurrence, the signature, and that the result must not depend on the worker count.

## Results

```
results/runs.jsonl                 one RunRecord per line, every graded run
results/manifest-july-2026.tsv     the original run manifest, kept verbatim except for paths
results/manifest-phase2-july-2026.tsv   an identical copy; cite either
results/tokens-new-problems-july-2026.json   request and token counts per model
results/runs/<problem>/<backend>/<model>/<agent>/   SUMMARY.txt, bench.csv, the kernel and test the agent wrote
```

Serving facts for the July models, as recorded in `serving` and `quantisation`:
claude-sonnet-4.6, kimi-k2.5 and llama-3.3-70b were served by Amazon Bedrock through a
LiteLLM gateway, precision not disclosed by the provider; gemma-31b was served by vLLM
on one NVIDIA L40S with 48 GB, checkpoint and quantisation not recorded at the time.

## Writing rules

Plain, clear, concise, correct British English in full sentences: code comments,
docstrings, documents and commit messages. No invented shorthand or slang. Spell out an
abbreviation the first time it is used. The default endpoint everywhere is
`http://localhost:4000/v1`.
