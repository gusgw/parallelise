# Operations

This document explains how to deploy the tool, check that it is ready, find out why a
conversion failed, and what stops generated code from doing harm. The module facts come
from `docs/ARCHITECTURE.md`.

## Deploying with Docker

The `Dockerfile` has two stages. The `cpu` stage starts from a small Python 3.12 image,
adds the C compiler, `make` and Open MPI (an implementation of the Message Passing
Interface, the standard for programs that run as several cooperating processes), installs
the package, and runs as an ordinary user. The `cuda` stage starts from an NVIDIA
development image and is for the CUDA backend on a machine with a graphics card.

    make image                                     # builds the cpu stage as parallelise:cpu
    docker run --rm parallelise:cpu health
    mkdir -p out
    docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/out:/work" \
      -e PARALLELISE_GATEWAY_URL=http://host.docker.internal:4000/v1 \
      -e PARALLELISE_MODEL=local \
      -e PARALLELISE_RESULTS=/work/runs.jsonl \
      parallelise:cpu run integrate openmp --agent basic

The image's entry point is the `parallelise` command, so everything after the image name
is a sub-command. The work folders go to `/work`; mount a directory there if you want to
keep them. Without `--user`, the container runs as its own user, which has identifier
10001, so a mounted directory must be writable by that user, and the files written into it
belong to it. With `--user "$(id -u):$(id -g)"` the container runs as the calling user and
the files belong to that user; make the mounted folder yourself first, because Docker would
otherwise make it owned by the root user. The results file is put in the mounted folder as
well, since its default place is inside the image, where only the image's own user may
write. `scripts/smoke-test.sh` runs the image this way where Docker is available.

The settings come from environment variables: `PARALLELISE_GATEWAY_URL`,
`PARALLELISE_MODEL`, `PARALLELISE_TEMPERATURE`, `PARALLELISE_API_KEY`,
`PARALLELISE_TIMEOUT`, `PARALLELISE_TIME_LIMIT`, `PARALLELISE_THROTTLE_WAIT`,
`PARALLELISE_WORK_ROOT` and `PARALLELISE_RESULTS`. The framework versions that the image
installs are the exact ones pinned in `pyproject.toml`, from which the image is installed.

## Deploying with Kubernetes

`k8s/job.yaml` runs one conversion cell as a Job; `k8s/README.md` explains each field.
Push the image to a registry the cluster can reach, set `image:` and the gateway address
in the ConfigMap, then run `kubectl apply -f k8s/job.yaml`. To run another cell, change
the arguments and the Job name; to run many, make one copy of the Job per cell.

## Checking health before a run

    parallelise health

This calls the gateway's `/models` list, confirms that the configured model is in it,
makes one tiny chat request, and reports whether the gateway was reachable, which models
it offers and the round-trip time. It exits with status 1 if something is wrong, and the
message says what to do next. Run it before a long batch, since a wrong address or model
name otherwise only shows up as a failed run.

## Throttling, time limits and deleting

**Throttling.** A shared model service often refuses requests for a while when it is busy,
answering with HTTP (the protocol of the web) status 429, which means too many requests.
parallelise waits and asks again instead of giving up. The wait is the number of seconds
in the `Retry-After` header when the answer has one; otherwise it starts at 2 seconds and
doubles with each refusal, up to 30 seconds, and it is never less than 1 second. The waits
for one request add up to at most `PARALLELISE_THROTTLE_WAIT` seconds (default 120; 0 means
never wait). If a `Retry-After` is longer than what is left, or the allowance is spent, the
request fails and the message names this setting. Every wait is written to the log at
warning level ("the endpoint answered HTTP 429 ... waiting 4 seconds ..."). All three agent
designs wait in the same way, so that a throttled service does not favour one of them. A
header that gives a date instead of a number of seconds is ignored, and the doubling wait is
used. Waiting counts against the time limit below, but a wait is not cut short by it.

**The time limit.** One agent run may take at most `--time-limit` seconds of wall-clock time
(or `PARALLELISE_TIME_LIMIT`; default 1800, which is 30 minutes). Each agent design looks at
the clock before every call to the model and, once the limit has passed, stops there, between
steps. It does not interrupt a step already under way, so a run can pass the limit by the
length of its last model call, build command or wait. The check then judges what is on disk.
The record's failure stage is the check's stage, and the notes say "The time limit of N
seconds stopped the agent." The limit covers the agent only; the check has its own limit,
`--check-timeout` (default 300 seconds). The limit that applied is stored as `time_limit_s`.

**Deleting.** No command deletes a folder it did not make unless you add `--force`.
`parallelise assemble`, `parallelise study mask` and `parallelise run` accept a destination
that does not exist or is empty. If it holds files they stop, name the folder, and tell you
to choose another folder or add `--force`, which deletes what is there first. A path that is
a file is always refused. `run` makes one work folder for each cell (a problem on a backend
with a model and an agent design), at `<work root>/<problem>/<backend>/<model>/<agent>`, and a
masked copy of the problem at `<work root>/_masked/<problem>`. It writes a marker file,
`.parallelise-made`, into each. A later run of the same cell deletes and replaces a folder
that has the marker, with no option needed; a folder without it (for example one made by an
earlier version of the program, or by hand) is refused until you move it or add `--force`.

## Where the logs are and why a conversion failed

- Messages from the program go to the standard error stream, which is the container log
  (`docker logs`, or `kubectl logs job/<name>`). Logger names begin `parallelise.`.
- Every graded run is appended as one JSON line to the file named by `PARALLELISE_RESULTS`
  (default `results/runs.jsonl`). The field `failure_stage` says how far the run got:

  | `failure_stage`     | Meaning |
  |---------------------|---------|
  | `none`              | The run passed. |
  | `no-files`          | The agent did not write the routine file. |
  | `tampered`          | The agent changed a trusted file (the benchmark, a header, the Makefile or a helper), or added a file named `GNUmakefile` or `makefile`. The log names the files. |
  | `build`             | The routine did not compile with the problem's own files, or compiled with warnings, which count as errors. |
  | `test-one-worker`   | The problem's own unit test failed with a single worker. |
  | `test-many-workers` | It passed with one worker but failed with several; usually a race or a wrong partition of the work. |
  | `benchmark`         | The benchmark failed, or the speed-up figures were rejected or below the floor. |
  | `timeout`           | A step of the check took longer than its limit. |
  | `agent-error`       | Older records only: the agent or the model client raised an error, and no check was run. A new run never writes this stage; see below. |

  To list failures: `grep '"result": "FAIL"' results/runs.jsonl`.

  The check runs whatever became of the agent. If the agent raised an error, if the model
  service failed in the middle of the run, or if the time limit stopped the agent, the
  routine that is on disk is judged all the same, and the record holds the check's result and
  stage. The notes then carry a sentence that says what stopped the agent: "The agent stopped
  with" followed by the error, or "The time limit of 1800 seconds stopped the agent." So a
  run can pass although its agent ended in an error, if the routine written before the error
  is good. To find the runs whose agent was interrupted, search the notes:
  `grep -E 'agent stopped with|time limit of' results/runs.jsonl`.
- Each record also says how the run was made, so that two records can be compared fairly.
  The fields after `notes` were added later; a record written before they existed has `null`,
  an empty text or an empty mapping in their place, and the report tools read both kinds.

  | Field | Meaning |
  |-------|---------|
  | `temperature` | The sampling temperature the model was asked to use. |
  | `step_budget` | The most model steps the agent was allowed (`--step-budget`). |
  | `time_limit_s` | The wall-clock limit on the agent, in seconds. |
  | `agent_time_s` | Seconds from the start of the agent's run until it stopped. |
  | `check_time_s` | Seconds the check took afterwards. `wall_time_s` is the two together. |
  | `framework_versions` | The installed version of each of httpx, httpx2, pydantic-ai, langgraph, langchain, langchain-openai and openai, those that are installed. |
  | `cpu_count` | The number of processor cores the process could use, which limits the speed-ups the benchmark can show. |
  | `package_version` | The version of parallelise itself. |
  | `source_hash` | Twelve hexadecimal digits of a hash of every Python file in the installed package and of the C and shell helpers copied into work folders. Two records with the same value came from the same code, and any change to the code changes it. |

  The counts of `requests`, `prompt_tokens` and `completion_tokens` are kept when a run ends in
  an error: they cover every model call made before the failure, for all three agent designs.
  A count of zero in a record made before this was kept can mean "lost", not "none".
- The check's own log, which holds the compiler output, the test output, the benchmark
  table and one line about the agent's own test program, is written to `check.log` in the
  work folder, and is kept as evidence under
  `results/runs/<problem>/<backend>/<model>/<agent>/` next to `SUMMARY.txt` and
  `bench.csv`. For a failure at `build`, read the compiler messages there. For a rejected
  benchmark, `parallelise bench-check <csv>` prints the reasons.
- The check judges the routine in a fresh folder that it assembles in a temporary directory
  from the problem's own files, with the problem's own test program. When the check fails,
  that folder is kept as `check-fail` inside the work folder, with the build products and
  the benchmark table, so that you can see exactly what was judged. When the check passes,
  the folder is deleted. The agent's own test program is run only for the log, and its
  result never changes the verdict.
- To judge a work folder again without the agent: `parallelise check <work_folder>
  --problem <problem> --backend <backend>`. The first argument is a finished work folder,
  not a problem; `--problem` is the path to the problem folder it was built from, or the
  name of a shipped example, and for a masked run it is the masked copy that `run` keeps
  under `<work_root>/_masked/`. The older name `grade` still works and prints a notice.

## Running a campaign

A campaign is a named run of `scripts/run_campaign.sh`, which converts a set of problems with
several models and agent designs, each problem both under its own name and masked behind a
neutral alias (`problem_a`, `problem_b` and so on), one container at a time. The name is
required and has no default: set `CAMPAIGN` to a name that says what the campaign is and when
it ran, for example `CAMPAIGN=campaign-2026-10`, and set `PARALLELISE_GATEWAY_URL` to the
model gateway address.

    CAMPAIGN=campaign-2026-10 PARALLELISE_GATEWAY_URL=http://localhost:4000/v1 \
      scripts/run_campaign.sh

The campaign's folder is made by the script from the name: the store `results/$CAMPAIGN`
(under the folder named by `BASE`, which defaults to the current folder) holds `runs.jsonl`,
the log of every run and the check evidence, and the work folders go to `work/$CAMPAIGN`.
The script prints the store path when it starts. Running it again with the same name adds
to the same store, so use a new name for a new campaign. `MODELS`, `PROBLEMS` and `BACKENDS`
restrict the run to part of the design. Afterwards, `parallelise study report --results
results/$CAMPAIGN/runs.jsonl` prints the tables.

## The smoke test

`scripts/smoke-test.sh` tests the tool as a new user would meet it. It clones the
repository from the checkout that holds the script (or from `REPO`, if set) into a temporary
folder, creates a virtual environment, installs the package with the development and report
extras, and starts the scripted stand-in model from `tests/fake_model_server.py`. It then
runs `parallelise health`, converts `problems/integrate` for OpenMP with the plain agent,
runs `parallelise check` on the work folder, and prints the report tables. If Docker is
available it also builds the CPU image and runs `health` and the same conversion inside it;
otherwise it prints a notice and skips those steps. Every step prints what a good result
looks like, and the script stops with a non-zero status at the first failure. Because it
clones, it tests what has been committed, not uncommitted edits. The conversion includes the
speed-up check, so the machine needs at least two free cores. The workflow in
`.github/workflows/ci.yml` runs the script as the job `smoke` after the tests.

## Limits on untrusted generated code

The model's output is untrusted. Four layers limit what it can do.

1. The agent can only run build commands from a short allow-list, and it runs them as
   argument lists with no shell, so it cannot chain commands, redirect output or expand
   variables. The list holds no Python interpreter and no shell. Each command runs in a
   process group of its own, which is killed as a whole at the time limit, and with an
   environment from which every variable that may hold a secret (a name containing KEY,
   TOKEN, SECRET or PASSWORD, and every variable beginning `PARALLELISE_`) has been removed.
   The check gives the same clean environment to the routine it builds and runs.
2. Every subprocess, whether a build, a test, a benchmark or a notebook execution, has a
   timeout, and a model request has its own (`PARALLELISE_TIMEOUT`).
3. The container runs as a non-root user. Under Kubernetes the Job also drops all
   capabilities, forbids privilege escalation and mounts the root file system read-only,
   leaving only the scratch volume writable.
4. The Job has processor and memory limits and a hard deadline (`activeDeadlineSeconds`).

These limits contain mistakes and runaway programs. They are not a full sandbox: the code
is compiled and run inside the container and could use the network unless you forbid it.
`make` and the compilers can start other programs, and a program running as the same user
can read the harness's own process information, which holds the gateway key if it was set
as an environment variable. For stronger isolation use a cluster network policy that allows
only the gateway.

## Troubleshooting

- **`health` says the gateway is unreachable.** Check `PARALLELISE_GATEWAY_URL`. Inside a
  container `localhost` means the container itself, not your computer.
- **`health` says the model is not listed.** Compare `PARALLELISE_MODEL` with the names
  in the model list that `health` prints.
- **Authentication errors (HTTP 401 or 403).** Set `PARALLELISE_API_KEY`. Client errors
  other than 429 are not retried; server errors and connection failures are retried three
  times.
- **The log says "the endpoint answered HTTP 429 ... waiting".** The model service is
  throttling requests, and parallelise is waiting as described above. If runs still fail with
  a message about 429, lower the load on the gateway, or raise `PARALLELISE_THROTTLE_WAIT`.
- **A run's notes say "The agent stopped with ..." after long waits.** The model is slow or
  failing; raise `PARALLELISE_TIMEOUT` or lower the load on the gateway. The check still ran,
  so the record shows whether the routine on disk passed.
- **A notes entry says the time limit stopped the agent.** Raise `--time-limit` if you want
  runs to continue longer, bearing in mind that the limit is also what bounds the cost of a
  run that is going nowhere.
- **`parallelise: The folder ... already holds files` or `... was not made by 'parallelise
  run'`.** The command refused to delete a folder it did not make. Choose another folder (for
  `run`, another `--work-root`), move the old one away, or add `--force` if you are sure it
  can go.
- **`parallelise: The results file ... cannot be made` or `cannot be written to`.** `run`
  checks before it starts that the record can be saved. Set `--results` (or
  `PARALLELISE_RESULTS`) to a file in a folder that you may write to, such as the mounted
  folder under Docker.
- **`mpicc` or `mpirun` not found.** The MPI packages are missing. The image contains them;
  on your own machine install `openmpi-bin` and `libopenmpi-dev`. The MPI tests skip
  themselves when they are absent.
- **Open MPI says there are not enough slots.** The container has fewer cores than the
  test asks for. The image sets `OMPI_MCA_rmaps_base_oversubscribe=1`; set it yourself
  elsewhere.
- **Open MPI complains when run as the root user.** Run as an ordinary user, as the image
  does. Do not work around it.
- **Permission denied writing `/work`.** The mounted directory is not writable by the user
  that the container runs as. Run the container with `--user "$(id -u):$(id -g)"` and make the
  folder yourself first, or, without `--user`, change the folder's owner or group permissions
  so that user 10001 may write to it.
- **Pod killed with reason `OOMKilled` (out of memory).** The memory limit was exceeded; raise it in
  `k8s/job.yaml` if the problem needs it.
- **Job failed with `DeadlineExceeded`.** The Job reached `activeDeadlineSeconds`.
- **CUDA tests skip.** There is no `nvcc` or graphics card, which is expected on the CPU
  image. Use the `cuda` stage on a machine with a driver and the NVIDIA container toolkit.
- **`make check-references` fails.** A known-good solution no longer passes its check. Run
  the failing test alone with `pytest -q -k <problem>` and read the check log.

## What continuous integration does and does not prove

The GitHub Actions workflow runs the linter, the type checker, the whole test suite with
every reference check for OpenMP and MPI, a build of the CPU container image, and the smoke
test, which repeats the documented steps from a fresh clone and inside Docker. It runs on
shared test servers with a few busy processor cores, whose timings mean nothing, so the
workflow sets `PARALLELISE_JUDGE_SPEED=0`. With that setting the check still builds each
routine with warnings treated as errors, runs the harness's test at one worker and at
several, and runs the benchmark program, so continuous integration proves that the code
works; but it does not judge the speed-up, and the tests that are about the speed rules
themselves are skipped. Speed is judged on a real multi-core machine with nothing else
running, where the variable is left unset. CUDA checks skip wherever there is no graphics
card.

## Memory when running benchmarks in containers

The MPI benchmarks for the prefix sum, decay and reservoir problems allocate two arrays of
eight bytes per element on every rank. At the default length of 134,217,728 elements that
is 2 GiB per rank, and the sweep runs as many ranks as the machine has cores, so a
12-core host needs 24 GiB for one benchmark. Three such benchmarks at once exhausted the
memory of a 30 GB host during the October 2026 campaign, and the kernel killed unrelated
services. Run every container with an explicit memory limit and no swap, for example
`docker run --memory=8g --memory-swap=8g`, run one benchmark container at a time, check
that at least 12 GB is free before starting, and size the problem to the limit with
`BENCH_N` and `BENCH_MAX_RANKS` rather than raising the limit. A container killed for
memory exits with status 137.
