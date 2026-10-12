# parallelise

This repository holds an agent that turns a numerical algorithm written in plain, serial
Python into parallel C or CUDA. It uses OpenMP for threads within one computer, MPI (the
Message Passing Interface) for processes that exchange messages, or CUDA for a graphics
processor, whichever suits the problem. Given a research notebook and a backend, the agent
decides how to parallelise the work: how to split it, which pattern of reduction or
exchange to use, and how to keep the answer the same however many workers run. It then
writes, builds and tests the program until an independent gate accepts the result as
correct and faster. The goal is an agent that also chooses the method for each algorithm,
or recommends well-written serial code when an algorithm resists parallelisation. A first
version of that choice exists: `parallelise assess` asks a model for a reasoned verdict on
a notebook, and `parallelise run` can act on the verdict. These verdicts have not yet been
tested against measured results at scale; that is the next experiment. A large language
model drives the agent through one OpenAI-compatible gateway, so several models can be
plugged into the same system and compared. The results below measure how well different
models and different agent harnesses do the job: four models in each of two studies,
three ways of building the agent, three backends, nine algorithms, and a test of how much
of a model's success is memory of textbook kernels rather than reasoning.

This repository holds my own work. It is a public version: material that belongs in
private repositories was removed before publication, and the history begins at
publication.

## What it does and why

A few words recur below. The kernel is the C function that does the computation. A cell is
one problem on one backend for one model. The oracle is the correct answer, recorded by the
notebook itself. A work folder is the directory the agent writes into.

Researchers often have an algorithm that works in a Jupyter notebook, an interactive
document that mixes text and Python, but runs slowly because it runs on one core.
Rewriting it in parallel C takes skill, and a language model can draft it. The difficulty
is trust. A model can produce code that compiles, looks plausible, and is wrong, or is not
parallel at all. This project therefore treats the model as an untrusted author and builds
the checking around it.

A problem is a notebook whose last cell records its own answer, with a tolerance. The agent
receives a work folder holding the notebook, the oracle in C form, a header giving the
exact function to implement, and a benchmark that it cannot change. It writes the kernel
and a unit test, and it must make `make check` pass. The gate then judges the work on its
own, in three parts:

1. The code must compile with every warning treated as an error.
2. The unit test must match the notebook's answer within the tolerance, with one worker and
   with several. The answer must not depend on the number of workers; that is how a broken
   parallelisation shows itself.
3. The benchmark must show a real speed-up, and the figures are checked so that they cannot
   have been invented.

The target is one of three backends. OpenMP runs several threads inside one computer. MPI
runs several processes that exchange messages. CUDA, NVIDIA's system for programming
graphics processors, runs the work on a graphics processing unit (GPU). Every graded run
becomes one structured record, so that models and agent designs can be compared afterwards.

### Choosing the backend

Before any code is written, `parallelise assess <notebook>` gives the whole notebook to a
model and asks for a critical analysis: which loop iterations are independent, whether
arithmetic or memory limits the work, what a distributed version would have to
communicate, and which of OpenMP, MPI, CUDA, a combination of them with a backend for each
part, or well-written serial code suits the algorithm. The command prints the analysis and
the recommendation, and `--out verdict.json` saves the verdict. The command
`parallelise run <problem> auto` makes the same assessment first and then runs the
recommended backend, provided the problem has a folder for it. If the verdict is serial
code, or names a backend the problem lacks, the command stops with a plain message and
exit status 1, and writes no run record. Otherwise it saves the verdict as
`assessment.json` in the work folder and mentions it in the run's notes. The verdicts from
July 2026 are in `results/assessments/july-2026/`.

The study asks three questions. Which model writes the best parallel code? Does the way the
agent is built matter? And how much of a strong model's success is memory of famous
textbook algorithms rather than reasoning? The last question is why two problems were built
to resist recall, and why a masking tool renames the familiar ones.

## Results

Every number below comes from a single run, not an average. The full tables, each with the
file it came from, are in [docs/RESULTS.md](docs/RESULTS.md); the sources are the run
manifests in `results/`. Here a cell means one problem on one backend for one model, scored
by how many of the three agent designs passed it. Single runs carry no error bars, so a
difference of one run between two cells lies within the noise.

### The seven well-known problems

The seven problems are all well known: numerical integration, matrix multiply, a heat
stencil (the Jacobi method), N-body forces, conjugate gradient, prefix sum and histogram.
Each model attempted each of them with each of three agent designs, on every backend that
suits it; CUDA was not built for prefix sum or histogram. The table counts passing runs.

| model | OpenMP (of 21) | MPI (of 21) | CUDA (of 15) | all (of 57) |
|---|---|---|---|---|
| claude-sonnet-4.6 | 21 | 21 | 15 | 57 |
| kimi-k2.5 | 21 | 20 | 15 | 56 |
| gemma-31b | 21 | 17 | 15 | 53 |
| llama-3.3-70b | 12 | 8 | 6 | 26 |

A chart of the same data is in [docs/figures/pass-rates.svg](docs/figures/pass-rates.svg).

Two things stand out. The two strongest models pass almost everything, so these problems
cannot tell them apart. And the difficulty lies almost wholly in MPI, where processes must
exchange boundary data correctly; every model but the weakest does well with OpenMP
threads. My project log says that a later uniform re-grade did not confirm Llama's one pass
on prefix sum with MPI. That re-grade is not in the run records, so the table counts the
pass; see [docs/RESULTS.md](docs/RESULTS.md).

### The recall-resistant problems

The seven problems above are famous, so a strong model may be remembering a known answer.
Two further problems have no standard parallel recipe. In `decay`, each step takes the
largest of three quantities, so the terms combine by maximum and addition instead of the
usual multiply and add. In `reservoir`, a tank level can neither overflow nor run empty, so
the tempting shortcut of adding everything up and clamping at the end gives the wrong
answer. Both run on OpenMP and MPI. Each entry below counts passing runs out of six: three
agent designs on each of two backends.

| model | decay | reservoir | total (of 12) |
|---|---|---|---|
| claude-sonnet-4.6 | 6 | 6 | 12 |
| gemma-31b | 6 | 5 | 11 |
| kimi-k2.5 | 4 | 4 | 8 |
| llama-3.3-70b | 0 | 0 | 0 |

**These numbers come from my project log, not from run records, and cannot be checked.**
The run files for these two problems were not kept. If the log is right, these problems
spread the models across the whole range, which the textbook problems could not; I hold
that conclusion loosely. Gemma's score above Kimi's does not make Gemma the better model:
Kimi's four failures were near misses, correct with one worker and slightly wrong at the
seams between the blocks given to different workers.

### Does the agent design matter?

The three designs give the model the same tools. They differ in how firmly they insist that
the checks pass before the model may finish. The plain agent, `basic`, is a short loop that
runs the checks itself when the model stops and feeds the failures back. The LangGraph
agent does the same in whole attempts. The Pydantic AI agent, `pydantic`, uses an output
validator that refuses the model's final answer until the checks pass.

| agent design | strong models (Sonnet, Kimi), of 38 | weak models (Gemma, Llama), of 38 |
|---|---|---|
| basic | 37 | 30 |
| langgraph | 38 | 25 |
| pydantic | 38 | 24 |

In these single runs the differences between models dwarf the differences between designs.
The designs hint at a pattern: strictness probably helps a capable model that makes careless
first attempts, and probably hurts a model that cannot solve the problem, because that model
then retries until its budget runs out. On the recall-resistant problems the strict agent
passed all four of Kimi's runs where the other two passed two of four each. That is a lean
seen in single runs, not an established effect. The reasoning is in
[docs/DESIGN.md](docs/DESIGN.md).

### Masked against unmasked (October 2026)

To measure recall directly, a second campaign ran integrate, jacobi, scan and conjugate
gradient on OpenMP and MPI, once as they stand and once after the masking tool had hidden
every name that gives the algorithm away. It used four models (Claude Sonnet 4.5, Kimi
K2.5, gpt-oss-120b and Llama 3.3 70B) and three agent designs, one run per cell, 192 runs
in all. The table gives passes out of 24 runs per model.

| model | unmasked | masked |
|---|---|---|
| claude-sonnet-4.5 | 24 | 21 |
| kimi-k2.5 | 24 | 20 |
| gpt-oss-120b | 21 | 20 |
| llama-3.3-70b | 7 | 9 |
| all models (of 96) | 76 | 70 |

For the three capable models together, the pass rate fell from 96% to 85% when the names
were hidden. The fall is modest, it falls mostly on prefix sum and on jacobi under MPI, and
single runs cannot tell it from chance. The new gate also rejected four runs in which an
agent had rewritten files it must not touch, which the July gate could not have caught. The
method, every table, the latency and cost figures and the limitations are in
[docs/CAMPAIGN-2026-10.md](docs/CAMPAIGN-2026-10.md); the chart is
[docs/figures/masked-vs-unmasked.svg](docs/figures/masked-vs-unmasked.svg).

### Measured token costs

A proxy between the agents and the gateway counted every token, and the counts were priced
at list prices. This campaign covered ten combinations of problem and backend, each
attempted by three agent designs; the source is `results/tokens-new-problems-july-2026.json`.

| model | requests | tokens sent plus received | cost |
|---|---|---|---|
| kimi-k2.5 | 260 | 1.12 million | about $0.83 |
| claude-sonnet-4.6 | 355 | 2.47 million | about $8.80 |
| gemma-31b | 763 | 8.93 million | about $5 to $6 of rented instance time |
| llama-3.3-70b | 1,944 | 39.9 million | about $28.75 |

The weakest model cost the most. It probably kept retrying problems it could not solve, and
each retry sends the whole growing conversation again. A weak model is not a cheap model.
Costs per agent design were not recorded, so this explanation rests on the failure records
and not on billing data.

## How to run it

You need Python 3.11 or later, a C compiler (`gcc`), `make`, and Open MPI (`mpicc` and
`mpirun`; on Debian or Ubuntu the packages are `openmpi-bin` and `libopenmpi-dev`). CUDA
needs `nvcc` and a graphics card, and its tests skip themselves when either is absent. You
also need an OpenAI-compatible gateway; LiteLLM is one way to provide one.

1. Install the package. The plain agent needs only `httpx`; the extras add the two
   framework agents, the chart and the development tools.

        git clone https://github.com/gusgw/parallelise.git parallelise && cd parallelise
        python3 -m venv .venv && . .venv/bin/activate
        make install

2. Point it at your gateway and check that it answers.

        export PARALLELISE_GATEWAY_URL=http://localhost:4000/v1
        export PARALLELISE_MODEL=kimi-k2.5
        export PARALLELISE_API_KEY=...        # only if your gateway needs one
        parallelise health

   This lists the models on the gateway, confirms that yours is among them, and makes one
   tiny request. If anything is wrong it exits with status 1 and a plain message.

3. Check the machinery without any model. This builds the known-good solution of every
   problem and runs the full gate on it.

        make check-references

4. Run one conversion: one problem, one backend, one agent. The run writes a work folder
   under `work/`, or wherever `PARALLELISE_WORK_ROOT` points, grades it with the gate, and
   appends a record to `results/runs.jsonl`.

        parallelise run integrate openmp --agent basic --model kimi-k2.5

5. Judge a work folder again, without the agent.

        parallelise grade <work folder> --backend openmp

6. Make a masked copy of a problem, with neutral names and no revealing words, for the
   masking study.

        parallelise mask scan kernel_a masked/kernel_a

7. Print the result tables and draw the chart from the stored records.

        parallelise report --store results/campaign-2026-10/runs.jsonl \
          --prices results/campaign-2026-10/prices.json \
          --masked-chart docs/figures/masked-vs-unmasked.svg

   Without a store it reads `results/runs.jsonl`. When the store holds masked and unmasked
   runs, the report adds the comparison tables; when it holds request counts, it adds a
   serving table, with cost if you pass a price file.

With Docker, the image holds the compilers and Open MPI and runs as an ordinary user:

    make image
    docker run --rm parallelise:cpu health
    docker run --rm -v "$PWD/out:/work" \
      -e PARALLELISE_GATEWAY_URL=http://host.docker.internal:4000/v1 \
      -e PARALLELISE_MODEL=kimi-k2.5 \
      parallelise:cpu run integrate openmp --agent basic

A Kubernetes job for one conversion is in `k8s/job.yaml`. Deployment, logs and
troubleshooting are described in [docs/OPERATIONS.md](docs/OPERATIONS.md).

## The design, for the specialist

[docs/DESIGN.md](docs/DESIGN.md) sets out the experiment: the three controlled axes of
model, agent design and backend; the frozen agent versions recorded with every run; the
three-part gate and why each part exists; the checks that stop an agent inventing its
speed-ups, and the incident that prompted them; the rule that the answer must not depend on
the worker count; the benchmark warm-up fix; the recall-resistant problems and the masking
study; the finding about strict agents on strong and weak models; and the costs and
limitations. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) maps the code.

## What is open source and what is my own

I used these from others:

- The models. Claude Sonnet 4.6 is proprietary and was used through a service. Kimi K2.5,
  Llama 3.3 70B and Gemma 31B have open weights, each under its own licence.
- The agent frameworks Pydantic AI and LangGraph, with LangChain beneath LangGraph.
- LiteLLM, which provides the gateway; vLLM, which serves a model on a graphics card; and
  Amazon Bedrock, a commercial service that hosts several of the models.
- The standard tools `gcc`, Open MPI, OpenMP, CUDA, nanobind and httpx.

I wrote these, and they are all in this repository:

- The harness: the oracle extractor, the assembly of work folders, the gate, the model
  client and the results store.
- The nine problems: notebooks, oracles, kernel headers, trusted benchmarks, Makefiles and
  known-good reference solutions.
- The benchmark checker that rejects invented speed-ups.
- The masking tool that makes recall-resistant copies of problems.
- The three agents, including the plain agent written directly on `httpx`.
- The nanobind wrappers that call a finished kernel from Python, and the design of the
  experiments and their analysis.

## How the models were served

One LiteLLM gateway stood between the agents and the four models of the July study. The
October campaign replaced two of them: Claude Sonnet 4.5 and gpt-oss-120b, both served by
Amazon Bedrock, took the places of Sonnet 4.6 and Gemma 31B.

- Amazon Bedrock served `claude-sonnet-4.6`, `kimi-k2.5` and `llama-3.3-70b`. The provider
  does not disclose the numerical precision, so the quantisation is recorded as "not
  disclosed".
- vLLM served `gemma-31b` on one NVIDIA L40S graphics card with 48 GB of memory. The
  checkpoint and the quantisation were not written down at the time, so they are unknown.

Where the information does not exist, the run records say "not recorded" or "not
disclosed". Nobody should compare these results with other work on the assumption of full
precision. The July runs did not record wall-clock time or latency; those fields
(`wall_time_s`, `latency_mean_s` and `tokens_per_second`) are empty in all 237 July records.
The October campaign filled them for every one of its 192 runs. The medians over runs were
1.1 to 3.8 seconds per call and 58 to 207 completion tokens per second, with
claude-sonnet-4.5 the slowest and gpt-oss-120b the fastest. The table, with request and
token totals and cost, is in [docs/CAMPAIGN-2026-10.md](docs/CAMPAIGN-2026-10.md). These
are latencies through a gateway to a shared cloud service on one day, not properties of the
models.

## Security

Code written by a model is untrusted. Four layers limit the harm it can do, and none of
them is a sandbox. The agent can run only commands from a short allow-list, as argument
lists with no shell; but the list includes the Python interpreter, so a determined agent
can leave the work folder, and the container, not the allow-list, is the boundary that
matters. The gate compares the trusted files, which are the benchmark, the headers and the
Makefile, against digests taken when the work folder was made, and fails the run if the
agent changed any of them; the digests live in the work folder, so this catches careless
edits rather than deliberate ones. Every subprocess, whether a build, a test, a benchmark
or a notebook run, has a timeout, and so does every model request. The container runs as
an ordinary user; under Kubernetes it drops all capabilities, has a read-only file system,
and runs under processor and memory limits and a hard deadline. These limits contain
mistakes and runaway programs. They do not make a sandbox, and the code could use the
network unless you forbid it, for example with a network policy; see
[docs/OPERATIONS.md](docs/OPERATIONS.md).

## Status and next steps

The comparison of four models, three agent designs and three backends on seven problems is
complete, and so is the first run on the two recall-resistant problems. The
masked-against-unmasked campaign of October 2026 is also done: its 192 records are in
`results/campaign-2026-10/runs.jsonl`, the 237 July records in `results/runs.jsonl` hold no
masked run, and the campaign gave the first measured latency and speed. Repeated runs, which
would put error bars on single-run differences, and a sweep of the sampling temperature
remain to be done. [docs/DESIGN.md](docs/DESIGN.md) lists the limitations that call for
them.

On 11 October 2026 every reference solution on OpenMP, MPI and CUDA, and a masked MPI
reference, passed the gate on a 12-core host with a 16 GB NVIDIA graphics card; see
[results/reference-gates-gpu-host.md](results/reference-gates-gpu-host.md).

<!-- statement on the use of AI tools: to be added by the author -->

## Licence and author

The MIT licence applies; see [LICENSE](LICENSE). Angus Gray-Weale wrote this.
