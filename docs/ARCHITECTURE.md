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
reproduce. `ORACLE` holds the parameters, the expected values and the tolerance, and
optionally the terms of a sum. The notebook computes the expected values itself, and the
terms let the extractor measure how far a correct program may legitimately differ by adding
them in another order.

A **routine** is the one C or CUDA function that does the computation, with exactly the
signature that the problem's header states. The agent writes it, together with a small test
of its own for its own use; the check judges the routine with the benchmark and with the
problem's own test, both of which call it. The routine, its files and the notebook are
named after the problem folder. (CUDA uses the word kernel for a function that runs on the
graphics card; under CUDA the routine is the function that starts such kernels. This note
says routine for the function the agent writes under every backend, and uses kernel only in
that CUDA sense.)

A **model** is a large language model reached through the gateway by name, for example
`kimi-k2.5`. The harness sends it the conversation and the tool definitions and receives
text or tool calls; it knows nothing else about the model.

An **agent** is the program that drives a model through one conversion. It hands the model
tools to read the notebook, read and write files and run build commands, keeps the
conversation going until the model says it has finished, runs `make check` in the work
folder itself at that point, and feeds failures back until a step budget or a time limit
runs out. An **agent design** is one way of building that program. There are three: a plain loop written on `httpx`, one built on
Pydantic AI, and one built on LangGraph. They give the model the same tools and differ in
how firmly they refuse to let it finish before the checks pass.

A **backend** is the parallel method: OpenMP (Open Multi-Processing, threads inside one
computer), MPI (Message Passing Interface, cooperating processes that exchange messages) or
CUDA (Compute Unified Device Architecture, NVIDIA's system for graphics cards).

A **cell** is one problem on one backend with one model and one agent design. A **work
folder** is the flat directory that `parallelise` builds for an agent to write into. It
holds the notebook, the oracle in C form, a header giving the exact signature to implement,
a benchmark program that the agent cannot change, and a Makefile. The agent works there and
runs `make check` there to test its own work, but the verdict is not taken from that folder;
the check takes only the routine out of it.

The **check** is the independent judgement of a finished work folder (`check.py`). It runs
after the agent has finished, without the model, and has the same form for every problem.
It judges one thing the agent wrote, the routine, and trusts nothing else in the work
folder. It does not build or run the agent's own test program to reach its verdict, and it
uses none of the work folder's trusted files (the Makefile, the benchmark, the headers and
the helpers). Instead it assembles a **judging folder** in a temporary directory from the
problem's own files, in the way a work folder is assembled, so that every trusted file comes
from the problem and none from the work folder. Into that folder it copies two files and no
others: the agent's routine (`<name>.c`, or `<name>.cu` for CUDA) and the problem's own
known-good test program, `reference/test_<name>.c`, which the agent never sees. That test
program calls the routine and compares its answer with the expected values in `reference.h`.
The oracle supplies only the expected values and the tolerance, which the harness writes
into `reference.h` as C constants, and a small helper, `testing.h`, compares numbers within a
tolerance and counts failures. A routine that computes the wrong answer therefore fails,
however weak or empty the agent's own test program is.

Before the four stages, the check looks for broken rules in the work folder. A run fails at
the stage called `tampered` if any trusted file in the work folder differs from the
fingerprint written when the work folder was assembled (kept in `.trusted.json`), if any
trusted file differs from the copy in the freshly assembled judging folder (which catches an
agent that rewrote `.trusted.json` to match its edits), or if the work folder holds a file
named `GNUmakefile` or `makefile`, which `make` reads in preference to `Makefile`. This
stage does not protect the verdict, which already comes from the judging folder; it exists
so that an agent which breaks the rules fails. The log names each file. If the routine
file is missing, the run fails at the stage called `no-files`. The four stages follow in
order, and the check stops at the first failure:

1. Build the routine, the problem's test program and the benchmark program in the judging
   folder, with every compiler warning treated as an error.
2. Run the problem's test program with one worker: one thread for OpenMP, one process for
   MPI. It must report no failures, which means that its assertions found the routine's
   answer within the oracle's tolerance.
3. Run the same test program again with several workers. It must again report no failures.
   A parallelisation whose answer depends on the number of workers fails here. The
   parallelism is in the routine under test, not in the test: there is one test program,
   run twice, and the routine inside it splits its work across the workers it is given.
4. Run the benchmark program, which times the routine at several worker counts and writes a
   table, and judge the table by the rules against invented or merely serial speed-ups,
   including a floor on the best speed-up.

A run that fails is recorded with the stage at which it failed. The check always runs, and
always judges the routine that is on disk when the agent stops, however the agent stopped: when
it declared itself finished, when it spent its step budget, when the time limit stopped it,
and when it raised an error. In the last two cases the record holds the check's result and
stage, and the notes say what stopped the agent. When the check fails, the
judging folder is kept as `check-fail` inside the work folder, so that a reader can see
exactly what was judged, with the compiler output, the binaries and the benchmark table;
when the check passes, the judging folder is deleted. The mathematics of the problem does
not change the form of the check, only its numbers: what is compared (a single number for
integration, a checksum of an array for the stencil and the prefix sum); the tolerance,
which the notebook sets from how much the result can legitimately move when the terms are
summed in another order, and which is zero for the integer-exact problems; the speed-up
floor, lower for problems limited by memory bandwidth than for those limited by arithmetic;
and the benchmark size.

The agent's own test program plays no part in the verdict. For the record only, after the
verdict is settled, the check builds that program against the agent's routine in a second
fresh folder, with the problem's Makefile, runs it with one worker, and adds one line to the
log saying whether it built and passed, built and failed, did not build, did not finish
within a minute, was not run for want of time, or did not exist. That line never changes the verdict. A run that stops at `tampered`
or `no-files` has no such line.

## Layout

```
README.md                 what it does, results for a general reader, how to run it
LICENSE                   MIT
pyproject.toml            package metadata, dependencies (the framework versions are pinned), settings
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
  (default 600), `time_limit_s` (default 1800: the wall-clock limit on one agent run),
  `throttle_wait_s` (default 120: the most seconds to wait, in total, for one request that is
  refused with HTTP 429), `work_root` (default `work`), `results_path` (default
  `results/runs.jsonl`). Environment variables: `PARALLELISE_GATEWAY_URL`,
  `PARALLELISE_MODEL`, `PARALLELISE_TEMPERATURE`, `PARALLELISE_API_KEY`,
  `PARALLELISE_TIMEOUT`, `PARALLELISE_TIME_LIMIT`, `PARALLELISE_THROTTLE_WAIT`,
  `PARALLELISE_WORK_ROOT`, `PARALLELISE_RESULTS`. The older
  names `MODEL_BASE_URL`, `MODEL_NAME`, `MODEL_TEMPERATURE` and `MODEL_API_KEY` are
  accepted as fallbacks. Validation raises `ConfigError` with a plain message: the URL
  must start with http or https, the temperature must lie in [0, 2], the timeout and the
  time limit must be positive, the throttle wait must not be negative, and the API key must
  be plain ASCII text, since it is sent in an HTTP header. `without_secrets(environ=None)` returns a copy of the environment without
  any variable whose name contains KEY, TOKEN, SECRET or PASSWORD (in any capitals) and
  without any variable whose name begins `PARALLELISE_`; the agents' tools and the check
  give that copy to every program they start.
- `log.py` — `configure_logging(level: str, json_lines: bool = False)`.
- `model.py` — `ModelClient(settings)` with `chat(messages, tools) -> ChatResult`
  and `health_check() -> HealthReport`. `ChatResult` holds `message` (the assistant
  message dict), `prompt_tokens`, `completion_tokens`, `latency_s`. The client keeps a
  list of `CallRecord(latency_s, prompt_tokens, completion_tokens, status)` and a
  `summary()` returning requests, total tokens, mean and median latency, and
  completion tokens per second of generation time. `health_check` calls `/models`,
  confirms the configured model is listed, makes one tiny chat call, and reports
  reachability, the model list, and the round-trip time. Retries: three attempts with
  back-off on connection errors and HTTP 5xx; none on other 4xx. An HTTP 429 answer (too many
  requests) is not an attempt used up but is waited out: the wait is the number of seconds in
  the `Retry-After` header when it holds one (a date is ignored), and otherwise doubles from 2
  seconds with each answer up to 30 seconds; no wait is shorter than 1 second. The waits for
  one request add up to at most `throttle_wait_s`; a `Retry-After` longer than what is left
  ends the waiting at once. Each wait is logged at warning level, and when the allowance is
  spent `ModelError` says so and names `PARALLELISE_THROTTLE_WAIT`. `Throttle` holds this
  rule. `ThrottledClient` and `ThrottledAsyncClient` (and `throttled_async_client_class`,
  which makes the same class from another library's asynchronous client) apply it inside
  the HTTP clients that the two framework agents hand to their frameworks. They send a refused
  request again from within `send`, so that the request and response hooks which time requests
  run once per attempt, and they mark the last 429 answer with `x-should-retry: false` so
  that the OpenAI library under both frameworks does not repeat the waiting.
- `results.py` — `RunRecord` dataclass and `ResultsStore(path)` with `append(record)`
  and `load() -> list[RunRecord]`, one JSON object per line. Fields of `RunRecord`:
  `when` (ISO 8601), `problem`, `backend`, `model`, `agent`, `agent_version`
  (short hash of the agent source), `masked` (bool), `result` ("PASS" or "FAIL"),
  `failure_stage` (one of `none`, `no-files`, `tampered`, `build`, `test-one-worker`,
  `test-many-workers`, `benchmark`, `timeout`, `agent-error`; the last is kept so that
  older records load, and a new run never writes it, because a run whose agent raised an
  error is judged by the check like any other), `speedup_best`
  (float or null), `wall_time_s` (from the start of the agent's run to the end of the check),
  `requests`, `prompt_tokens`, `completion_tokens`,
  `latency_mean_s`, `tokens_per_second`, `serving` (provider identifier or
  "not recorded"), `quantisation` (string or "not disclosed"), `notes`.
  Fields added later, each with a default so that older records still load (the default
  means "not recorded"): `temperature`, `step_budget`, `time_limit_s` (the limit that applied
  to the agent), `agent_time_s` and `check_time_s` (seconds, kept apart), `framework_versions`
  (a mapping from package name to installed version, for httpx, httpx2, pydantic-ai,
  langgraph, langchain, langchain-openai and openai, whichever are installed), `cpu_count`
  (processor cores the process could use), `package_version` (the version of parallelise) and
  `source_hash` (twelve hexadecimal digits of a hash of every `.py` file of the installed
  package and of the C and shell helpers in `data/`, so that any code change shows). Defaults:
  `None` for the numbers, an empty text for the two strings, an empty mapping for the
  versions. `from_dict` ignores keys it does not know and reads a null `framework_versions`
  as empty, so records written by other versions load. `ResultsStore.ensure_writable()`
  raises `ResultsError` if a record could not be added, without creating anything; `run`
  calls it before the agent starts. `append` and `load` turn an unwritable or unreadable
  file, or one that is not UTF-8 text, into `ResultsError`.
- `provenance.py` — `framework_versions()`, `package_version()`, `cpu_count()` and
  `source_hash(package_dir=None)`, which supply the last five fields above. The core count is
  the number of cores the process may run on (`os.sched_getaffinity`), falling back to
  `os.cpu_count()`.
- `folders.py` — the rules about deleting. `prepare_destination(dest, force)` accepts a
  destination that is absent or empty and refuses one that holds files, with a message that
  names the folder and `--force`; with `force` it deletes the folder; a path that is a file is
  always refused. `prepare_own_folder(folder, force)` is for the folders that `run` makes: it
  deletes one that carries the marker file `.parallelise-made` (written by `write_marker` when
  `run` made it), and refuses any other folder that holds files unless `force` is given.
  `ensure_folder(folder, what)` makes a folder or says plainly why it cannot.
- `oracle.py` — `extract_oracle(notebook: Path) -> Oracle` executes the notebook's
  code cells in a fresh namespace with a timeout and reads `ORACLE`; validates that
  every expected value is finite and that, where `terms` are given, their sum
  reproduces the expected value within tolerance; `write_reference(oracle, dest)`
  writes `reference.json` and `reference.h`. It rejects an oracle whose values are not finite or whose terms do not sum to the
  expected value.
- `assemble.py` — `assemble_work_dir(problem_dir, backend, dest, include_reference=False,
  force=False)` builds a flat work folder (`dest` must be absent or empty unless `force` is
  true; the problem is examined first, so a bad problem never costs the destination): package data files, the notebook, `reference.*`,
  `problem.h` if present, the backend's header, `bench.c` and `Makefile`, and the record of
  fingerprints `.trusted.json`; with `include_reference=True` it also copies the known-good
  solution and test, which is how the machinery is tested without a model. The check calls
  it to make the judging folder. `AssembleError`, raised for an unusable problem folder, is a
  `ParalleliseError`, so the command line shows its message plainly.
- `check.py` — `run_check(work_dir, problem_dir, backend, timeout_s=300, speedup_floor=None)
  -> CheckResult(passed, failure_stage, speedup_best, log)`. The `problem_dir` is the problem
  folder that the work folder was assembled from (the masked copy for a masked run). The
  function first looks for broken rules in the work folder (fingerprints, the fresh copy,
  `GNUmakefile` and `makefile`) and fails at the stage `tampered` if it finds any. It then
  assembles the judging folder with `assemble_judging_folder` (the problem's files plus the
  agent's routine plus the problem's known-good test), builds with the backend's compiler,
  runs the problem's test at one worker and at several, runs the benchmark and calls
  `bench_check`, all inside the judging folder. The folder is deleted on success and kept as
  `check-fail` in the work folder on failure. Last, for the log only, it builds and runs the
  agent's own test program at one worker. `require_reference_test(problem_dir, backend)`
  raises `AssembleError` if the problem has no known-good test, and `run` calls it before the
  agent starts. If the check cannot run at all (the computer will not give it a temporary
  folder, for instance), `run_check` raises `CheckError`, which is the only use of that class;
  a routine that fails is a result and not an error. The code under judgement runs with an environment from which every variable
  that may hold a secret has been removed. The floor is the least best speed-up that the
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
  problem must still pass its reference check. `mask_problem(problem_dir, dest, alias,
  force=False)` refuses a destination that holds files unless `force` is true. Every problem
  with its arguments (an alias that is not an identifier or names an algorithm, a folder
  without exactly one notebook, a notebook that is not valid JSON, a destination that cannot be
  written) is found before anything is deleted and raised as `InputError`.
- `agents/base.py` — `Agent` protocol: `run(work_dir: Path, task: str, settings:
  Settings, client: ModelClient, step_budget: int) -> AgentResult(finished,
  steps, requests, prompt_tokens, completion_tokens, latencies_s, notes,
  time_limit_reached, error)`. `Deadline(seconds)` is the wall-clock limit: each agent makes
  one from `settings.time_limit_s` when its run starts and asks `expired()` before every call
  to the model, so that a run past its limit stops cleanly between steps. A step already
  under way is not interrupted, so a run may overrun by the length of its last step, and a
  wait for a throttled request (see `model.py`) is not cut short. A run stopped this way
  returns `time_limit_reached=True`. A failure of the model service or the framework ends the
  run too, but is returned as `error` (the exception's name and message) with the counts of
  the requests made before it, instead of being raised; an agent that does raise is also
  handled by `run`. Counts are therefore kept when a run fails. `agents/__init__.py` exposes `get_agent(name)` for `basic`,
  `pydantic` and `langgraph`; the two framework agents import their frameworks lazily
  and raise `AgentUnavailable` with an installation hint if the extra is missing.
  All three share one tool surface (`agents/tools.py`): read the notebook, read a file,
  write a file, run an allow-listed command. The allow-list holds `make`, `gcc`, `cc`,
  `clang`, `mpicc`, `mpirun`, `nvcc`, `ls` and `cat`, and programs inside the work folder
  written as `./name`; it holds no interpreter, so `python3`, `bash` and `sh` are refused,
  and a `./` path that leaves the work folder is refused. Every command runs without a
  shell, with its input closed, in a new session so that it and its children form one
  process group, and with the secret-free environment from `config.without_secrets`. After
  the same time limit as before (600 seconds) the whole process group is killed with
  `SIGKILL`, so that no child survives. All three run `make check` in the work folder
  themselves when the model says it has finished, and feed failures back, but they stop
  differently. That command uses the work folder's own Makefile and the agent's own test
  program and is the agent's guide only; the verdict comes from `check.py`. All three also stop
  when the time limit has passed. `basic` stops when the
  checks pass or after `step_budget` model calls (default 80). `langgraph` makes up to
  `MAX_PASSES` (8) passes, each limited to `step_budget` steps. `pydantic` stops when the
  checks pass or when the framework's request limit, set to `step_budget`, is reached; it
  drives the framework's graph one node at a time (`agent.iter`) so that it can check the time
  before each model call and read the usage after a failure, and it reaches the model through
  an `httpx2` client, which is what Pydantic AI asks for (the older `httpx` client is
  deprecated there and warns). The tests treat a notice that something will be removed in
  version 3 as an error. `langgraph` streams each pass state by state and stops, between
  states, when the time limit has passed and the next step would call the model.
  `docs/DESIGN.md` explains the differences.
- `study/report.py` — `build_tables(records, masked=False, prices=None) -> str` (Markdown: pass
  counts by model and backend, by agent design, by problem; when the records hold masked and
  unmasked runs, masked-against-unmasked tables; when they hold request counts, a serving
  table with cost if `prices` is given), `load_prices(path)`, `build_chart(records,
  dest_svg)` (grouped bars of pass rate by model and backend) and
  `build_comparison_chart(records, dest_svg)` (pass rate by model, unmasked beside masked),
  both drawn with matplotlib. `parallelise study report` takes `--store`, `--prices`, `--chart` and
  `--masked-chart`. The report reads only the fields it needs, so the fields added to records
  later change nothing in its tables, and older records, which lack them, are read as before. A
  timestamp that is not a date, a numeric field that holds text, or a price file that is missing,
  is not JSON, has the wrong shape or lacks a price raises `ResultsError`, whose message names
  the record or the file. A chart that cannot be drawn or written is reported in one line, with
  an installation hint when matplotlib is missing.
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
  `mask` and `report` as hidden aliases that print a notice and run the new form. `mask` takes
  `--force`.
- `cli.py` — `main(argv=None)` using argparse. The commands listed by `parallelise --help`
  are `health`, `assess <notebook> [--out verdict.json]`, `run <problem> <backend> [--agent]
  [--model] [--step-budget] [--time-limit SECONDS] [--check-timeout] [--force]`, `check
  <work_folder> --problem <problem> --backend <backend>` and `study`.
  The argument named `<problem>` is the path to a problem folder (a notebook with the files
  for each backend) or the name of one of the shipped examples under `problems/`;
  `<work_folder>` is a finished work folder, not a problem. `check` needs the problem as
  well as the work folder because it rebuilds the folder from the problem's own files and
  judges only the agent's routine, with the problem's own test; for a masked run the
  problem is the masked copy, which `run` keeps under `<work_root>/_masked/`. `run` passes
  the check the problem folder it assembled from, and refuses to start an agent on a
  problem that has no known-good test. Before the agent starts, `run` also checks that the
  results file can be written and that the work root can be made, and refuses a step budget
  below 1. It runs the agent, then the check, whatever became of the agent, and records the
  agent's time and the check's time apart (`wall_time_s` is their sum, and the setting-up
  before the agent is not counted). If the agent raised an error or returned one, or the time
  limit stopped it, the notes say so and the failure stage is the check's. Counts come from
  the agent's result, and from the model client's own record when the agent kept none, so an
  agent that raised an error after making calls still reports them. `assess` is a separate
  command and `run` never calls it. Three maintenance commands work but are not listed:
  `oracle <notebook>`, `assemble <problem> <backend> <dest> [--force]` and `bench-check
  <csv>`. `study mask` also takes `--force`. The old name `grade` for `check` is a
  hidden alias that prints a notice. Exit status is 0 on success, 1 on a failed check or a
  reported error, and 2 on bad arguments.

### Rules for errors and for deleting

Every error that a user can cause is a `ParalleliseError` (or a subclass) with a message of one
line that says what went wrong and what to do; the command line prints that line after
`parallelise:` and returns status 1, and prints no stack trace. The subclasses are
`ConfigError` (a setting), `InputError` (a name, a path, a folder or a file that the user
gave; it is also a `ValueError`, so that older callers still catch it), `AssembleError`,
`OracleError`, `ModelError`, `ResultsError` (records, the results file and price files),
`AgentUnavailable` and `CheckError`. Code that reads or writes a path the user gave wraps an
operating-system error in one of these, naming the path. As a last net, `main` reports any
other `OSError` that reaches it as one line with the path and the words "Check the path and its
permissions". A programming error still shows its trace.

No command deletes a folder it did not make without `--force`. `assemble`, `study mask` and
`run` refuse a destination that holds files, naming the folder and the option. `run` replaces
its own earlier work folder for the same cell, and its masked copy under `<work_root>/_masked/`,
only if the folder has the marker file `.parallelise-made`; work folders made before the marker
existed need `--force` once.

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

The files in `reference/` are never copied into an agent's work folder. The check copies
`test_<name>.c` (plain C under every backend, CUDA included) into the judging folder to
judge the agent's routine, and the machinery tests copy both files to prove that a correct
solution passes.

The routine, its files and the notebook are named after the problem folder. In the folder
`integrate`, the notebook is `integrate.ipynb`, the header is `integrate.h`, the routine is
written in `integrate.c` (`integrate.cu` for CUDA) and its test in `test_integrate.c`. The
header gives the name and signature of the function itself, which may be longer than the
folder name, as in `nbody_accel` for the folder `nbody`. `ORACLE` holds the parameters, the expected values and the tolerance, and
optionally the terms of a sum; it does not name the routine.

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
