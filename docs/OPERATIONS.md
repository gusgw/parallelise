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
    docker run --rm -v "$PWD/out:/work" \
      -e PARALLELISE_GATEWAY_URL=http://host.docker.internal:4000/v1 \
      -e PARALLELISE_MODEL=local \
      parallelise:cpu run integrate openmp --agent basic

The image's entry point is the `parallelise` command, so everything after the image name
is a sub-command. The work folders go to `/work`; mount a directory there if you want to
keep them. The container's user has identifier 10001, so a mounted directory must be
writable by it.

The settings come from environment variables: `PARALLELISE_GATEWAY_URL`,
`PARALLELISE_MODEL`, `PARALLELISE_TEMPERATURE`, `PARALLELISE_API_KEY`,
`PARALLELISE_TIMEOUT`, `PARALLELISE_WORK_ROOT` and `PARALLELISE_RESULTS`.

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

## Where the logs are and why a conversion failed

- Messages from the program go to the standard error stream, which is the container log
  (`docker logs`, or `kubectl logs job/<name>`). Logger names begin `parallelise.`.
- Every graded run is appended as one JSON line to the file named by `PARALLELISE_RESULTS`
  (default `results/runs.jsonl`). The field `failure_stage` says how far the run got:

  | `failure_stage`     | Meaning |
  |---------------------|---------|
  | `none`              | The run passed. |
  | `no-files`          | The agent did not write the kernel or the unit test. |
  | `build`             | The code did not compile, or compiled with warnings, which count as errors. |
  | `test-one-worker`   | The unit test failed with a single worker. |
  | `test-many-workers` | It passed with one worker but failed with several; usually a race or a wrong partition of the work. |
  | `benchmark`         | The benchmark failed, or the speed-up figures were rejected or below the floor. |
  | `timeout`           | A step took longer than its limit. |
  | `agent-error`       | The agent or the model client raised an error, for example the gateway was unreachable. |

  To list failures: `grep '"result": "FAIL"' results/runs.jsonl`.
- The gate's own log, which holds the compiler output, the test output and the benchmark
  table, is kept as evidence under `results/runs/<problem>/<backend>/<model>/<agent>/`
  next to `SUMMARY.txt` and `bench.csv`. For a failure at `build`, read the compiler
  messages there. For a rejected benchmark, `parallelise bench-check <csv>` prints the
  reasons.
- To judge a work folder again without the agent: `parallelise grade <work_dir> --backend
  <backend>`.

## Limits on untrusted generated code

The model's output is untrusted. Four layers limit what it can do.

1. The agent can only run build commands from a short allow-list, and it runs them as
   argument lists with no shell, so it cannot chain commands, redirect output or expand
   variables.
2. Every subprocess, whether a build, a test, a benchmark or a notebook execution, has a
   timeout, and a model request has its own (`PARALLELISE_TIMEOUT`).
3. The container runs as a non-root user. Under Kubernetes the Job also drops all
   capabilities, forbids privilege escalation and mounts the root file system read-only,
   leaving only the scratch volume writable.
4. The Job has processor and memory limits and a hard deadline (`activeDeadlineSeconds`).

These limits contain mistakes and runaway programs. They are not a full sandbox: the code
is compiled and run inside the container and could use the network unless you forbid it.
For stronger isolation use a cluster network policy that allows only the gateway.

## Troubleshooting

- **`health` says the gateway is unreachable.** Check `PARALLELISE_GATEWAY_URL`. Inside a
  container `localhost` means the container itself, not your computer.
- **`health` says the model is not listed.** Compare `PARALLELISE_MODEL` with the names
  in the model list that `health` prints.
- **Authentication errors (HTTP 401 or 403).** Set `PARALLELISE_API_KEY`. Client errors
  are not retried; server errors and connection failures are retried three times.
- **Runs end in `agent-error` after long waits.** The model is slow; raise
  `PARALLELISE_TIMEOUT` or lower the load on the gateway.
- **`mpicc` or `mpirun` not found.** The MPI packages are missing. The image contains them;
  on your own machine install `openmpi-bin` and `libopenmpi-dev`. The MPI tests skip
  themselves when they are absent.
- **Open MPI says there are not enough slots.** The container has fewer cores than the
  test asks for. The image sets `OMPI_MCA_rmaps_base_oversubscribe=1`; set it yourself
  elsewhere.
- **Open MPI complains when run as the root user.** Run as an ordinary user, as the image
  does. Do not work around it.
- **Permission denied writing `/work`.** The mounted directory is not writable by user
  10001. Change its owner or its group permissions.
- **Pod killed with reason `OOMKilled`.** The memory limit was exceeded; raise it in
  `k8s/job.yaml` if the problem needs it.
- **Job failed with `DeadlineExceeded`.** The Job reached `activeDeadlineSeconds`.
- **CUDA tests skip.** There is no `nvcc` or graphics card, which is expected on the CPU
  image. Use the `cuda` stage on a machine with a driver and the NVIDIA container toolkit.
- **`make check-references` fails.** A known-good solution no longer passes its gate. Run
  the failing test alone with `pytest -q -k <problem>` and read the gate log.

## What continuous integration does and does not prove

The GitHub Actions workflow runs the linter, the unit tests, every reference gate for
OpenMP and MPI, and a build of the CPU container image on every push. The hosted runner
has only a few shared processor cores, so a genuinely parallel kernel often shows no
speed-up there. The reference gates on the runner therefore prove correctness: the code
builds with warnings treated as errors and gives the right answer at one worker and at
several. A reference that is correct but misses the speed-up floor is reported as skipped,
with the measured speed-up in the message, because the environment variable
`PARALLELISE_SPEEDUP_UNRELIABLE=1` is set in the workflow. The speed-up requirement is
checked on a multi-core host, where that variable is left unset and a missed floor is a
failure. CUDA gates skip wherever there is no graphics card.
