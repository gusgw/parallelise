# parallelise

parallelise turns a numerical algorithm written as serial Python in a Jupyter notebook into
a parallel C or CUDA program and checks the result. An agent driven by a large language
model writes the program for a chosen backend: OpenMP for threads within one computer, MPI
(the Message Passing Interface) for processes that exchange messages, or CUDA for a
graphics processor. The agent decides how to parallelise the work: how to split it, which
pattern of reduction or exchange to use, and how to keep the answer the same however many
workers run. It writes, builds and tests the program until an independent check accepts
the result as correct and faster than the serial original. A separate command asks a
model which method suits a notebook, or whether the algorithm is better left serial and
well written. The models are reached through one OpenAI-compatible gateway, so any model
served that way can be used and several can be compared. The measurements made so far, on
nine algorithms with four models and three ways of building the agent, are in
[docs/RESULTS.md](docs/RESULTS.md).

This repository holds my own work. It is a public version: material that belongs in
private repositories was removed before publication, and the history begins at
publication.

## How it works

A **notebook** is a Jupyter notebook whose last cell defines a dictionary named `ORACLE`
holding the algorithm's parameters, the answer the notebook itself computed, and a
tolerance. That answer is the **oracle**: the C program must reproduce it within the
tolerance. The **routine** is the one C or CUDA function that does the computation, with
exactly the signature the problem's header states. (CUDA uses the word kernel for a
function that runs on the graphics card; this document uses routine throughout.)

A **problem** is a folder holding the notebook and, for each backend it supports, a header
with the routine's signature, a benchmark program and a Makefile. Nine problems are
shipped under `problems/`: numerical integration, matrix multiply, a heat stencil (the
Jacobi method), N-body forces, conjugate gradient, prefix sum, histogram, and two problems
built to have no textbook parallel recipe, a decaying score that combines terms by maximum
and addition, and a tank level that can neither overflow nor run dry.

The **agent** drives the model through one conversion. It hands the model tools to read the
notebook, read and write files, and run build commands; keeps the conversation going until
the model says it has finished; runs the checks itself at that point; and feeds any failure
back until a step budget runs out. There are three **agent designs**, which give the model
the same tools and differ in how firmly they refuse to let it finish before the checks
pass: a plain loop written on `httpx`, one built on Pydantic AI, and one built on
LangGraph.

Each conversion runs in a **work folder** that the harness assembles: the notebook, the
oracle written as C constants, the header, the benchmark program, the Makefile and a small
test helper. The agent writes two files into it, the routine and a test program for the
routine. The **check** then judges the folder after the agent has finished, without the
model, in four stages, stopping at the first failure:

1. Build the routine, the test program and the benchmark program with every compiler
   warning treated as an error.
2. Run the test program with one worker. It must report no failures, which means its
   assertions found the routine's answer within the oracle's tolerance.
3. Run the same test program with several workers. The answer must not depend on the
   number of workers.
4. Run the benchmark program, which times the routine at several worker counts, and judge
   its table: a genuine speed-up must reach the problem's floor, and the figures must look
   measured rather than invented.

The benchmark program, the header and the Makefile are trusted files: the check compares
them with fingerprints taken when the folder was assembled and refuses to judge a folder
in which any has changed. Every judged run is written as one record to a results store, one
JSON object per line: the problem, backend, model and agent design, the outcome and the
stage of any failure, the best speed-up, and the requests, tokens, latency and throughput of
the model calls.

## How to use it

You need Python 3.11 or later, a C compiler (`gcc`), `make`, and Open MPI (`mpicc` and
`mpirun`; on Debian or Ubuntu the packages are `openmpi-bin` and `libopenmpi-dev`). CUDA
needs `nvcc` and a graphics card, and its tests skip themselves when either is absent. You
also need an OpenAI-compatible gateway that serves the chosen model; LiteLLM is
one way to provide one.

1. Install the package. The plain agent needs only `httpx`; the extras add the two
   framework agents, the charts and the development tools.

        git clone https://github.com/gusgw/parallelise.git parallelise && cd parallelise
        python3 -m venv .venv && . .venv/bin/activate
        make install

2. Point it at the gateway and the model, and confirm that both answer.

        export PARALLELISE_GATEWAY_URL=http://localhost:4000/v1
        export PARALLELISE_MODEL=kimi-k2.5
        export PARALLELISE_API_KEY=...        # only if the gateway needs one
        parallelise health

   This lists the models the gateway serves, confirms the chosen one is among them, and
   makes one small request. It exits with status 1 and a plain message if anything is
   wrong.

3. Check the machinery without any model. This builds the known-good solution of every
   shipped problem and runs the full check on it.

        make check-references

4. Convert one problem on one backend with one agent design. The argument is the path to a
   problem folder, or the name of a shipped example under `problems/`. The run assembles a
   work folder under `work/` (or under `PARALLELISE_WORK_ROOT`), runs the agent, judges the
   result with the check, and appends a record to `results/runs.jsonl` (or to
   `PARALLELISE_RESULTS`).

        parallelise run integrate openmp --agent basic

5. Judge a finished work folder again, without the agent or the model.

        parallelise check work/integrate/openmp/kimi-k2.5/basic --backend openmp

6. Ask the model which method suits a notebook. The reply is a reasoned verdict, printed
   and optionally saved as JSON: which loop iterations are independent, whether arithmetic
   or memory limits the work, what a distributed version would have to communicate, and
   which of OpenMP, MPI, CUDA, a combination by part, or serial code it recommends.

        parallelise assess problems/jacobi/jacobi.ipynb --out verdict.json

7. The study tools live under one command. `study mask` makes a copy of a problem with a
   neutral name and every revealing word removed, for measuring how much a model's success
   is recall of a known algorithm; `study report` prints the result tables from a results
   store and draws the charts.

        parallelise study mask scan problem_c masked/problem_c
        parallelise study report --results results/campaign-2026-10/runs.jsonl \
          --prices results/campaign-2026-10/prices.json \
          --masked-chart docs/figures/masked-vs-unmasked.svg

Every command answers `--help` with one sentence per argument and an example.

With Docker, the image holds the compilers and Open MPI and runs as an ordinary user:

    make image
    docker run --rm parallelise:cpu health
    docker run --rm -v "$PWD/out:/work" \
      -e PARALLELISE_GATEWAY_URL=http://host.docker.internal:4000/v1 \
      -e PARALLELISE_MODEL=kimi-k2.5 \
      parallelise:cpu run integrate openmp --agent basic

A Kubernetes job for one conversion is in `k8s/job.yaml`. Deployment, logs,
troubleshooting and the memory rules for running benchmarks in containers are in
[docs/OPERATIONS.md](docs/OPERATIONS.md). The script `scripts/smoke-test.sh` runs the
steps above from a fresh clone against a scripted stand-in for the model, and continuous
integration runs it on every push.

## Adding a problem

A new problem is a folder with the notebook and, for each backend, the three files the
shipped problems have: a header giving the routine's exact signature, a benchmark program
that builds inputs of a chosen size, calls the routine at several worker counts after an
untimed warm-up call, and writes a timing table, and a Makefile whose `check` target builds
the routine and the test program, runs the test with one worker and with several, runs the
benchmark and judges its table against the problem's speed-up floor. The notebook's last
cell defines `ORACLE` with the parameters, the expected values and the tolerance, and may
hand over the terms of any sum so that the tolerance can be set from how far a correct
program may legitimately differ by adding them in another order. The shipped problems are
the models to copy; `problems/integrate` is the smallest. Generating the per-backend files
from the notebook alone is planned and not yet done.

## Results

Nine problems, three backends, four models and three agent designs have been run, in a
July 2026 study of 237 judged runs and an October 2026 campaign of 192 that compared each
problem as shipped with a copy whose revealing names had been removed. The tables, with
the file each came from, are in [docs/RESULTS.md](docs/RESULTS.md); the October campaign,
with its method, latency, throughput and cost, is in
[docs/CAMPAIGN-2026-10.md](docs/CAMPAIGN-2026-10.md); the charts are
[docs/figures/pass-rates.svg](docs/figures/pass-rates.svg) and
[docs/figures/masked-vs-unmasked.svg](docs/figures/masked-vs-unmasked.svg). All runs are
single attempts.

## Design

[docs/DESIGN.md](docs/DESIGN.md) sets out the experiment: the controlled axes of model,
agent design and backend; the frozen agent versions recorded with every run; the four-stage
check and why each stage exists; the rules that stop an agent inventing its speed-ups; the
requirement that the answer not depend on the worker count; the problems built to resist
recall and the masking study; and the limitations. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
maps the code: the conversion path in the package's top level and the study tools in the
`study` subpackage, each with its own tests.

## What is open source and what is my own

Used from others:

- The models. Claude Sonnet is proprietary and was used through a service. Kimi K2.5,
  Llama 3.3 70B, Gemma 31B and gpt-oss-120b have open weights, each under its own licence.
- The agent frameworks Pydantic AI and LangGraph, with LangChain beneath LangGraph.
- LiteLLM, which provides the gateway; vLLM, which serves a model on a graphics card; and
  Amazon Bedrock, a commercial service that hosts several of the models.
- The standard tools `gcc`, Open MPI, OpenMP, CUDA, nanobind and httpx.

My own, all in this repository:

- The harness: the oracle extractor, the assembly of work folders, the check, the model
  client, the assessment stage and the results store.
- The nine problems: notebooks, oracles, routine headers, trusted benchmarks, Makefiles and
  known-good reference solutions.
- The benchmark checker that rejects invented speed-ups, and the masking tool.
- The three agents, including the plain agent written directly on `httpx`.
- The nanobind wrappers that call a finished routine from Python, and the design of the
  experiments and their analysis.

## How the models were served

In the July 2026 study, Amazon Bedrock served `claude-sonnet-4.6`, `kimi-k2.5` and
`llama-3.3-70b`, and vLLM served `gemma-31b` on one NVIDIA L40S graphics card with 48 GB of
memory; all four were reached through one LiteLLM gateway. In the October 2026 campaign
Amazon Bedrock served `claude-sonnet-4.5`, `kimi-k2.5`, `gpt-oss-120b` and
`llama-3.3-70b`. Bedrock does not disclose the numerical precision of its models, and the
Gemma checkpoint and quantisation were not recorded; the run records say "not disclosed" or
"not recorded" in those fields. The July runs did not record latency or wall-clock time;
every October record carries requests, tokens, mean latency and completion tokens per
second. Those are latencies through a gateway to a shared cloud service on one day, not
properties of the models.

## Security

Code written by a model is untrusted. The harness limits what it can do in four ways, none
of which is a sandbox. The agent can run only commands from a short allow-list, as argument
lists with no shell; the list includes the Python interpreter, so a determined agent can
leave the work folder, and a container, not the allow-list, is the boundary that matters.
The check compares the trusted files against fingerprints taken when the work folder was
assembled and refuses to judge a folder in which any has changed; the fingerprints live in
the work folder, so this catches careless edits rather than deliberate ones. Every
subprocess, whether a build, a test, a benchmark or a notebook run, has a timeout, and so
does every model request. The container runs as an ordinary user; under Kubernetes it
drops all capabilities, has a read-only file system, and runs under processor and memory
limits and a hard deadline. The generated code can reach the network unless the deployment
forbids it, for example with a network policy; [docs/OPERATIONS.md](docs/OPERATIONS.md)
describes the settings.

## Licence and author

The MIT licence applies; see [LICENSE](LICENSE). Angus Gray-Weale wrote this.
