# parallelise

This system integrates several large language models, reached through one
OpenAI-compatible gateway, into a code-generation and verification pipeline. Given a serial
research notebook, an agent driven by a language model writes a parallel C program, and an
independent gate then checks that the program is correct and genuinely faster. Four models
are compared: Claude Sonnet 4.6 (`claude-sonnet-4.6`), Kimi K2.5 (`kimi-k2.5`), Llama 3.3
70B (`llama-3.3-70b`) and Gemma 31B (`gemma-31b`). Three ways of building the agent, and
three kinds of parallel hardware, are compared as well.

This repository is a public version containing the author's own work only, scrubbed of
proprietary material kept in private repositories, with a fresh history.

## What it does and why

Researchers often have an algorithm that works in a Jupyter notebook (an interactive
document that mixes text and Python) but runs slowly because it is serial. Rewriting it in
parallel C is skilled work, and a language model can draft it. The difficulty is trust: a
model can produce code that compiles, looks plausible, and is wrong or not actually
parallel. This project therefore treats the model as an untrusted author and builds the
checking around it.

A problem is a notebook whose last cell records its own answer, called the oracle, with a
tolerance. The agent receives a work folder containing the notebook, the oracle in C form,
a header with the exact function signature to implement, and a benchmark that it cannot
change. It writes the kernel and a unit test, and it must make `make check` pass. The gate
then judges the work on its own, in three parts:

1. The code must compile with every warning treated as an error.
2. The unit test must match the notebook's answer within the tolerance, both with one worker
   and with several. The answer must not depend on the number of workers, which is how a
   broken parallelisation is caught.
3. The benchmark must show a real speed-up, and the figures are validated so that they
   cannot have been invented.

The target hardware is one of three backends. OpenMP (Open Multi-Processing) uses several
threads inside one computer. MPI (the Message Passing Interface) uses several cooperating
processes that exchange messages. CUDA (NVIDIA's system for programming graphics
processors) uses a graphics processing unit, abbreviated GPU. Every graded run is stored as
one structured record, so that the models and agent designs can be compared afterwards.

The study asks three questions. Which model writes the best parallel code? Does the way
the agent is built matter? And how much of a strong model's success is memory of famous
textbook algorithms rather than reasoning? The last question is why two problems were built
to resist recall, and why a masking tool renames the familiar problems.

## Results

All numbers below are single runs, not averages. Full tables, with the file each came from,
are in [docs/RESULTS.md](docs/RESULTS.md). The sources are the run manifests in `results/`.

### The seven recallable problems

There are seven well-known problems: numerical integration, matrix multiply, a heat stencil
(the Jacobi method), N-body forces, conjugate gradient, prefix sum and histogram. Each was
attempted by each model with each of three agent designs, on every backend that suits it
(CUDA was not built for prefix sum or histogram). The table counts passing runs.

| model | OpenMP (of 21) | MPI (of 21) | CUDA (of 15) | all (of 57) |
|---|---|---|---|---|
| claude-sonnet-4.6 | 21 | 21 | 15 | 57 |
| kimi-k2.5 | 21 | 20 | 15 | 56 |
| gemma-31b | 21 | 17 | 15 | 53 |
| llama-3.3-70b | 12 | 8 | 6 | 26 |

A chart of the same data is in [docs/figures/pass-rates.svg](docs/figures/pass-rates.svg).

Two things stand out. The two strongest models pass almost everything, so these problems
cannot tell them apart. And the difficulty lies almost entirely in MPI, where processes
must exchange boundary data correctly; every model except the weakest does well with
OpenMP threads. The author's project log reports that a later uniform re-grade did not
confirm the one pass of Llama on prefix sum with MPI. That re-grade is not in the run
records, so the table counts the pass (see [docs/RESULTS.md](docs/RESULTS.md)).

### The recall-resistant problems

The seven problems above are famous, so a strong model may be recalling a known answer.
Two further problems have no standard parallel recipe. `decay` is a running score in which
each step takes the largest of three quantities, so it combines terms with maximum and
addition instead of the usual multiply and add. `reservoir` is a tank level that cannot
overflow or run empty, for which the tempting shortcut of adding everything up and clamping
at the end gives the wrong answer. Both run on OpenMP and MPI. Each entry below counts
passing runs out of six: three agent designs on each of the two backends.

| model | decay | reservoir | total (of 12) |
|---|---|---|---|
| claude-sonnet-4.6 | 6 | 6 | 12 |
| gemma-31b | 6 | 5 | 11 |
| kimi-k2.5 | 4 | 4 | 8 |
| llama-3.3-70b | 0 | 0 | 0 |

The run records for these two problems were lost with the host that ran them. The numbers
are quoted from the author's project log, written at the time, and cannot be re-checked
against records. The problems spread the models across the whole range, which the textbook
problems could not. Gemma scoring above Kimi should not be read as Gemma being the better
model: Kimi's four failures were near-misses, correct with one worker and slightly wrong at
the boundaries between the blocks given to different workers.

### Does the agent design matter?

The three designs give the model the same tools. They differ in how strictly they insist
that the checks pass before the model may finish. The plain agent (`basic`) is a short loop
that runs the checks itself when the model stops, and feeds failures back. The LangGraph
agent does the same in whole attempts. The Pydantic AI agent (`pydantic`) uses an output
validator that refuses to accept the model's final answer until the checks pass.

| agent design | strong models (Sonnet, Kimi), of 38 | weak models (Gemma, Llama), of 38 |
|---|---|---|
| basic | 37 | 30 |
| langgraph | 38 | 25 |
| pydantic | 38 | 24 |

Model quality matters about ten times more than agent design. Among designs, strictness
helps a capable model that makes careless first attempts, and hurts a model that cannot
solve the problem, because it then retries until its request budget is spent. On the
recall-resistant problems the strict agent passed all four of Kimi's runs where the other
two passed two of four each. The mechanism is explained in [docs/DESIGN.md](docs/DESIGN.md).

### Measured token costs

Tokens were counted exactly by a proxy placed between the agents and the gateway, and
priced at list prices. This campaign covered ten problem and backend combinations, each
attempted by three agent designs (source: `results/tokens-new-problems-july-2026.json`).

| model | requests | tokens sent plus received | cost |
|---|---|---|---|
| kimi-k2.5 | 260 | 1.12 million | about $0.83 |
| claude-sonnet-4.6 | 355 | 2.47 million | about $8.80 |
| gemma-31b | 763 | 8.93 million | about $5 to $6 of rented instance time |
| llama-3.3-70b | 1,944 | 39.9 million | about $28.75 |

The weakest model was the most expensive. When it cannot solve a problem it keeps
retrying, and each retry sends the whole growing conversation again. A weak model is not
a cheap model.

## How to run it

You need Python 3.11 or later, a C compiler (`gcc`), `make`, and Open MPI (`mpicc` and
`mpirun`; on Debian or Ubuntu the packages are `openmpi-bin` and `libopenmpi-dev`). CUDA
needs `nvcc` and a graphics card, and its tests skip themselves when they are absent. You
also need an OpenAI-compatible gateway; LiteLLM is one way to provide it.

1. Install the package. The plain agent needs only `httpx`; the extras add the two
   framework agents, the chart and the development tools.

        git clone https://github.com/gusgw/parallelise.git parallelise && cd parallelise
        python3 -m venv .venv && . .venv/bin/activate
        make install

2. Point it at your gateway and check that it works.

        export PARALLELISE_GATEWAY_URL=http://localhost:4000/v1
        export PARALLELISE_MODEL=kimi-k2.5
        export PARALLELISE_API_KEY=...        # only if your gateway needs one
        parallelise health

   This lists the models on the gateway, confirms yours is among them and makes one tiny
   request. It exits with status 1 and a plain message if something is wrong.

3. Check the machinery without any model. This builds the known-good solution of every
   problem and runs the full gate on it.

        make check-references

4. Run one conversion: one problem, one backend, one agent. The run writes a work folder
   (under `work/` unless `PARALLELISE_WORK_ROOT` says otherwise), grades it with the gate
   and appends a record to `results/runs.jsonl`.

        parallelise run integrate openmp --agent basic --model kimi-k2.5

5. Judge a work folder again, without the agent.

        parallelise grade <work folder> --backend openmp

6. Make a masked copy of a problem, with neutral names and no revealing words, for the
   masking study.

        parallelise mask scan kernel_a masked/kernel_a

7. Print the result tables, and draw the chart, from the stored records.

        parallelise report

With Docker, the image contains the compilers and Open MPI and runs as an ordinary user:

    make image
    docker run --rm parallelise:cpu health
    docker run --rm -v "$PWD/out:/work" \
      -e PARALLELISE_GATEWAY_URL=http://host.docker.internal:4000/v1 \
      -e PARALLELISE_MODEL=kimi-k2.5 \
      parallelise:cpu run integrate openmp --agent basic

A Kubernetes job for one conversion is in `k8s/job.yaml`. Deployment, logs and
troubleshooting are in [docs/OPERATIONS.md](docs/OPERATIONS.md).

## The design, for the specialist

[docs/DESIGN.md](docs/DESIGN.md) sets out the experimental design: the three controlled
axes (model, agent design, backend), the frozen agent versions recorded per run, the
three-part gate and why each part exists, the checks that stop an agent from inventing its
speed-ups (and the incident that motivated them), the requirement that the answer not
depend on the worker count, the benchmark warm-up fix, the recall-resistant problems and the
masking study, the finding about strict agents on strong and weak models, costs and
limitations. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) is the map of the code.

## What is open source and what is my own

Used from others:

- The models: Claude Sonnet 4.6 (a proprietary model, used through a service) and the
  open-weight models Kimi K2.5, Llama 3.3 70B and Gemma 31B, each under its own licence.
- The agent frameworks: Pydantic AI and LangGraph, with LangChain beneath LangGraph.
- LiteLLM, which provides the gateway; vLLM, which serves a model on a GPU; and Amazon
  Bedrock, a commercial service that hosts several of the models.
- Standard tools: `gcc`, Open MPI, OpenMP, CUDA, nanobind and httpx.

My own work, all in this repository:

- The harness: the oracle extractor, work-folder assembly, the gate, the model client and
  the results store.
- The nine problems: notebooks, oracles, kernel headers, trusted benchmarks, Makefiles and
  known-good reference solutions.
- The benchmark checker that rejects invented speed-ups.
- The masking tool that makes recall-resistant copies of problems.
- The three agents, including the plain agent written directly against `httpx`.
- The nanobind wrappers that call a finished kernel from Python, and the experimental
  design and its analysis.

## How the models were served

The four models were reached through one LiteLLM gateway.

- `claude-sonnet-4.6`, `kimi-k2.5` and `llama-3.3-70b` were served by Amazon Bedrock. The
  provider does not disclose the numerical precision, so the quantisation is recorded as
  "not disclosed".
- `gemma-31b` was served by vLLM on one NVIDIA L40S graphics card with 48 GB of memory. The
  checkpoint and the quantisation were not recorded at the time, so they are unknown.

Run records hold "not recorded" or "not disclosed" in these fields where the information
does not exist. Results should not be compared with other work on the assumption of full
precision. The July runs also did not record wall-clock time per run or latency;
measuring latency is under way (see below).

## Security

Code written by a model is untrusted. Four layers limit the harm it can do. The agent can
run only build commands from a short allow-list, as argument lists with no shell. Every
subprocess, whether a build, a test, a benchmark or a notebook run, has a timeout, and so
does every model request. The container runs as an ordinary user, and under Kubernetes it
drops all capabilities, has a read-only file system, and has processor and memory limits
and a hard deadline. These limits contain mistakes and runaway programs. They are not a full
sandbox, and the code could use the network unless you forbid it, for example with a
network policy; see [docs/OPERATIONS.md](docs/OPERATIONS.md).

## Status and next steps

The comparison of four models, three agent designs and three backends on seven problems is
complete, as is the first run on the two recall-resistant problems. In progress are the
masked campaign, which re-runs masked copies of familiar problems to measure how much of
the success is recall, and latency measurements of the gateway. Planned after those are
repeated runs, to put error bars on single-run differences, and a sweep of the sampling
temperature. [docs/DESIGN.md](docs/DESIGN.md) lists the limitations that motivate them.

## Licence and author

MIT licence; see [LICENSE](LICENSE). Written by Angus Gray-Weale.

<!-- statement on the use of AI tools: to be added by the author -->
