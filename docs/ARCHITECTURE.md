# Architecture and build contract

This document is the specialist's map of the code: the layout of the repository and the
interfaces between its parts. Every path below is relative to the repository root.

## What the system does

The program turns a numerical algorithm written as serial Python into a parallel C or CUDA
program, and judges the result independently. These words are used throughout.

A **notebook** is a Jupyter notebook, a file of cells holding Python code and prose. It is
the only form of input the harness accepts. The oracle extractor runs the notebook's code
cells in a child process and reads the dictionary named `ORACLE` that the last cell defines.

The **oracle** is the notebook's own answer with a tolerance, which the C code must
reproduce. The notebook computes the answer itself and may also hand over the terms of the
sum behind it, so that the extractor can measure how far a correct program may legitimately
differ by adding the terms in another order.

A **routine** is the one C or CUDA function that does the computation, with exactly the
signature that the problem's header states. The agent writes it, together with a small test
of it; the benchmark and the test both call it. (CUDA uses the word kernel for a function
that runs on the graphics card. This note says routine for the function the agent writes
under every backend, and uses kernel only in that CUDA sense.)

A **model** is a large language model reached through the gateway by name, for example
`kimi-k2.5`. The harness sends it the conversation and the tool definitions and receives
text or tool calls; it knows nothing else about the model.

An **agent** is the program that drives a model through one conversion. It hands the model
tools to read the notebook, read and write files and run build commands, keeps the
conversation going until the model says it has finished, runs the checks itself at that
point, and feeds failures back until a step budget runs out. An **agent design** is one way
of building that program. There are three: a plain loop written on `httpx`, one built on
Pydantic AI, and one built on LangGraph. They give the model the same tools and differ in
how firmly they refuse to let it finish before the checks pass.

A **backend** is the parallel method: OpenMP (Open Multi-Processing, threads inside one
computer), MPI (Message Passing Interface, cooperating processes that exchange messages) or
CUDA (Compute Unified Device Architecture, NVIDIA's system for graphics cards).

A **cell** is one problem on one backend with one model and one agent design. A **work
folder** is the flat directory that `parallelise` builds for an agent to write into and for
the check to judge. It holds the notebook, the oracle in C form, a header giving the exact
signature to implement, a benchmark program that the agent cannot change, and a Makefile.

The **check** is the independent judgement of a work folder (`check.py`). It runs after the
agent has finished, without the model, and has the same form for every problem. It judges
two files the agent wrote: the routine and a test program for it. The test program is the
agent's own; nothing in the oracle or the notebook specifies it. The oracle supplies only
the expected values and the tolerance, which the harness writes into `reference.h` as C
constants, and a small helper, `testing.h`, compares numbers within a tolerance and counts
failures. The agent decides what the test program calls and what it asserts. Before the
four stages, the check confirms that no trusted file has been changed since the work folder
was assembled; if one has, the run fails at the stage called `tampered`. The four stages
follow in order, and the check stops at the first failure:

1. Build the routine, the test program and the benchmark program, with every compiler
   warning treated as an error.
2. Run the test program with one worker: one thread for OpenMP, one process for MPI. It must
   report no failures, which means that its assertions found the routine's answer within
   the oracle's tolerance.
3. Run the same test program again with several workers. It must again report no failures.
   A parallelisation whose answer depends on the number of workers fails here. The
   parallelism is in the routine under test, not in the test: there is one test program,
   run twice, and the routine inside it splits its work across the workers it is given.
4. Run the benchmark program, which times the routine at several worker counts and writes a
   table, and judge the table by the rules against invented or merely serial speed-ups,
   including a floor on the best speed-up.

A run that fails is recorded with the stage at which it failed. The mathematics of the
problem does not change the form of the check, only its numbers: what is compared (a single
number for integration, a checksum of an array for the stencil and the prefix sum); the
tolerance, which the notebook sets from how much the result can legitimately move when the
terms are summed in another order, and which is zero for the integer-exact problems; the
speed-up floor, lower for problems limited by memory bandwidth than for those limited by
arithmetic; and the benchmark size. Because the test program is the agent's own, the check
trusts the agent to test honestly, within the limits that the stages above impose.

## Layout

```
README.md                 what it does, results for a general reader, how to run it
LICENSE                   MIT
pyproject.toml            package metadata, dependencies, pytest and ruff settings
Makefile                  convenience targets: install, lint, test, check-references, image
Dockerfile                CPU stage (gcc, OpenMPI, Python) and an optional CUDA stage
k8s/job.yaml              run one conversion cell as a Kubernetes Job
.github/workflows/ci.yml  lint, tests, reference checks, container build, smoke test
docs/ARCHITECTURE.md      this file
docs/DESIGN.md            the experimental design and what was learned
docs/RESULTS.md           every results table, with provenance
docs/OPERATIONS.md        deploying, health checks, reading the logs, troubleshooting
scripts/                  run_campaign.sh (a masked-against-unmasked campaign), smoke-test.sh
src/parallelise/          the package (see "Modules")
src/parallelise/study/    the study tools, apart from the conversion path: mask.py, report.py, cli.py
src/parallelise/data/     files copied into every work folder: testing.h, bench_sweep.sh
problems/<name>/          one folder per algorithm (see "Problems")
results/                  the run records and the per-run evidence (see "Results")
studies/coding-agents/    the earlier study of agent designs on a debugging task
bindings/                 nanobind wrappers that call the finished routines from Python
tests/                    pytest suite and the scripted model server (fake_model_server.py); `pytest -q` from the root runs it
tests/study/              tests of the study subpackage; `pytest -q tests/study` runs them alone
```

## Modules

All code is in `src/parallelise/`. The conversion path is the modules directly in that
folder, and the study tools are in the subpackage `src/parallelise/study/`. The study
tools are separate so that each part can be read and tested on its own: `pytest -q tests/study`
tests the study tools without the conversion path, and `pytest -q tests -k 'not study'` tests
the conversion path without them. The conversion path needs the study subpackage only when
`parallelise run --masked` calls the masking tool. Python 3.11 or later. Every public function has
type hints and a docstring in plain English. Logging goes through
`logging.getLogger("parallelise.<module>")` (`parallelise.study.<module>` for the study tools); nothing prints except the command-line
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
- `check.py` — `run_check(work_dir, backend, timeout_s=300, speedup_floor=None) ->
  CheckResult(passed, failure_stage, speedup_best, log)`. Before anything else it compares
  the trusted files (the benchmark, headers and Makefile) with the fingerprints written when
  the work folder was assembled, and fails at the stage `tampered` if any has changed. It
  then builds with the backend's compiler, runs the unit test at one worker and at several,
  runs the benchmark and calls `bench_check`. The floor is the least best speed-up that the
  table must reach. It comes from the problem's Makefile (`SPEEDUP_FLOOR`) and falls back to
  1.5 for OpenMP and MPI or 3 for CUDA; the module docstring explains it in full.
- `bench_check.py` — `check_cpu_table(path, floor) -> BenchVerdict(ok, reasons,
  speedup_best)` judges a table of times at several thread or process counts. It rejects a
  reported speed-up that disagrees with the measured times, duplicated widths, every
  parallel time exactly the serial time divided by the width, constant throughput,
  efficiency not equal to one at one worker, and a best speed-up under the floor.
  `check_gpu_table` judges the CUDA table, which has one row for the serial run and one for
  the graphics card. The older names `check_bench_csv` and `check_bench_cuda` remain as
  deprecated aliases.
- `study/mask.py` — `mask_problem(problem_dir, dest, alias) -> MaskReport`. Produces a copy
  of a problem whose routine, files, header guards, Makefile variables and benchmark
  refer to the neutral name `alias`; whose notebook markdown is replaced by one neutral
  sentence; whose code comments are removed; and whose identifiers that name the
  algorithm are renamed from a word list kept in the module. Numbers, logic and the
  oracle values are unchanged, and the report lists every substitution made. A masked
  problem must still pass its reference check.
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
- `study/report.py` — `build_tables(records, masked=False, prices=None) -> str` (Markdown: pass
  counts by model and backend, by agent design, by problem; when the records hold masked and
  unmasked runs, masked-against-unmasked tables; when they hold request counts, a serving
  table with cost if `prices` is given), `load_prices(path)`, `build_chart(records,
  dest_svg)` (grouped bars of pass rate by model and backend) and
  `build_comparison_chart(records, dest_svg)` (pass rate by model, unmasked beside masked),
  both drawn with matplotlib. `parallelise study report` takes `--store`, `--prices`, `--chart` and
  `--masked-chart`.
- `assess.py` — `assess_notebook(notebook: Path, client: ModelClient) -> Verdict` gives the
  whole notebook to the model and asks for an analysis of the dependency structure, the
  arithmetic intensity, the communication a distributed version would need, and a
  recommendation. `Verdict` holds `backend` (`openmp`, `mpi`, `cuda`, `serial` or
  `combination`), `parts` (for a combination, a list of `Part(part, backend)`), the reasoning
  fields, the raw `analysis` text, the `model` name and the call's tokens and latency. The
  JSON object at the end of the reply is found even when wrapped in prose or a code fence;
  if none is usable, `ParalleliseError` is raised. `Verdict.chosen_backend` is the single
  backend to run (for a combination, the first part that is not serial).
- `cli_support.py` — three small helpers used by both command-line modules: `lazy_import`
  (import a part on first use and turn a failure into a plain message), `resolve_problem`
  and `settings_from_args`.
- `study/cli.py` — `register(sub, common, problems)` adds the `study` command, with its
  subcommands `mask` and `report`, to the main parser. It also registers the old top-level
  `mask` and `report` as hidden aliases that print a notice and run the new form.
- `cli.py` — `main(argv=None)` using argparse. The commands listed by `parallelise --help`
  are `health`, `assess <notebook> [--out verdict.json]`, `run <problem> <backend> [--agent]
  [--model]`, `check <work_folder> --backend <backend>` and `study`. The argument named
  `<problem>` is the path to a problem folder (a notebook with the files for each backend)
  or the name of one of the shipped examples under `problems/`; `<work_folder>` is a
  finished work folder, not a problem. `assess` is a separate command and `run` never calls
  it. Three maintenance commands work but are not listed: `oracle <notebook>`, `assemble
  <problem> <backend> <dest>` and `bench-check <csv>`. The old name `grade` for `check` is a
  hidden alias that prints a notice. Exit status is 0 on success, 1 on a failed check or a
  reported error, and 2 on bad arguments. Every error message says what to do next.

## Problems

```
problems/<name>/<name>.ipynb      the notebook; its last cell defines ORACLE
problems/<name>/reference.json    generated by `parallelise oracle`; kept for reference
problems/<name>/reference.h       generated likewise
problems/<name>/problem.h         optional input helpers shared by all backends
problems/<name>/<backend>/<name>.h     the routine signature the agent must implement
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
results/runs/<problem>/<backend>/<model>/<agent>/   SUMMARY.txt, bench.csv, the routine and test the agent wrote
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
