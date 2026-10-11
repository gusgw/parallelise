# Results

Every results table in the project, with where it came from. The experimental design is in
`docs/DESIGN.md`.

## How to read the tables

- A **run** is one attempt by one agent design, with one model, on one problem and backend,
  judged by the gate. It passes or fails.
- A **cell** is one problem and backend for one model, scored out of the three agent designs.
  "2/3" means two of the three designs passed.
- All runs are single attempts. Where a cell was run more than once, the latest record is
  used.
- Backends: OpenMP (threads in one computer), MPI (cooperating processes), CUDA (a graphics
  processor).
- Model names are as the gateway knows them. In column headings "sonnet" is
  `claude-sonnet-4.6`, "kimi" is `kimi-k2.5`, "gemma" is `gemma-31b` and "llama" is
  `llama-3.3-70b`.

## Provenance

| source | what it holds | dates |
|---|---|---|
| `results/manifest-july-2026.tsv` | 237 graded run records: time, problem, backend, model, agent, result, short hash of the agent source | 4 and 5 July 2026 |
| `results/manifest-phase2-july-2026.tsv` | a duplicate of the first file, with identical content; "phase2" refers to the second stage of the campaign, which added the CUDA runs and the four harder problems | 4 and 5 July 2026 |
| `results/tokens-new-problems-july-2026.json` | exact request and token counts per model for the four harder problems | 5 and 6 July 2026 |
| my project log (not in this repository) | records of the campaigns whose files were not kept, the reference timings, and a later re-grade that the run records do not contain | 6 July 2026 |

The two manifests have identical content, so either can be cited. The 237 records cover 228 distinct cells of
problem, backend, model and agent: 108 for the first three problems and 120 for the four
harder ones. Six cells were run more than once, for nine extra records; seven of the
superseded records carry the note "A later run of the same cell replaced this one" in
`results/runs.jsonl`, and the other two (both Llama, histogram under MPI) do not. In all, 199
of the 237 records are passes. Every table in sections 1, 2, 4 and 5.1 to 5.4 was
recomputed from the manifest. The one respect in which they differ from my log is that none
applies the later re-grade of Llama's prefix sum under MPI.

Three kinds of numbers cannot be checked against a record: the results and costs for the
two recall-resistant problems (sections 3, 5.5, 5.6 and 6.2), the reference timings and
Python demonstrations (sections 7 and 8), and the re-grade of Llama's prefix sum under MPI.
All were written in my project log at the time, and the files behind the first group were
not kept. They are quoted from that log and are labelled as such.

## 1. The first three problems: 3 problems, 3 backends, 4 models

Problems: integration (the midpoint rule, which gives pi), matrix multiply and a heat
stencil (jacobi). Source: `results/manifest-july-2026.tsv`, 4 and 5 July 2026, 108 runs
under frozen agent versions. Each entry counts passes out of the three agent designs.

| problem and backend | sonnet | kimi | gemma | llama |
|---|---|---|---|---|
| integration, OpenMP | 3/3 | 3/3 | 3/3 | 3/3 |
| integration, MPI | 3/3 | 3/3 | 3/3 | 2/3 |
| integration, CUDA | 3/3 | 3/3 | 3/3 | 0/3 |
| matrix multiply, OpenMP | 3/3 | 3/3 | 3/3 | 3/3 |
| matrix multiply, MPI | 3/3 | 3/3 | 3/3 | 0/3 |
| matrix multiply, CUDA | 3/3 | 3/3 | 3/3 | 2/3 |
| heat stencil, OpenMP | 3/3 | 3/3 | 3/3 | 2/3 |
| heat stencil, MPI | 3/3 | 2/3 | 1/3 | 0/3 |
| heat stencil, CUDA | 3/3 | 3/3 | 3/3 | 2/3 |
| **total (of 27)** | **27** | **26** | **25** | **14** |

Notes. The OpenMP and MPI part was run first, with one frozen set of agents; its totals (of
18) were sonnet 18, kimi 17, gemma 16 and llama 10. OpenMP was nearly
uniformly easy, at 35 of 36 passes in that part; the one failure was Llama's Pydantic agent on
the heat stencil. The heat stencil under MPI, which needs boundary
exchange, was the cell that separated the models. One of Llama's CUDA failures on integration (the plain agent) was
a new kind: a correct answer but a speed-up of only 1.92 times, under the floor of 3 (the 1.92 is from my project log, not
from the records).

## 2. The four harder problems

Problems: N-body forces, conjugate gradient, prefix sum and histogram. Source:
`results/manifest-july-2026.tsv`, 5 July 2026, 120 runs. Prefix sum and histogram have no
CUDA version.

| problem and backend | sonnet | kimi | gemma | llama |
|---|---|---|---|---|
| N-body, OpenMP | 3/3 | 3/3 | 3/3 | 0/3 |
| N-body, MPI | 3/3 | 3/3 | 3/3 | 1/3 |
| N-body, CUDA | 3/3 | 3/3 | 3/3 | 2/3 |
| conjugate gradient, OpenMP | 3/3 | 3/3 | 3/3 | 2/3 |
| conjugate gradient, MPI | 3/3 | 3/3 | 1/3 | 1/3 |
| conjugate gradient, CUDA | 3/3 | 3/3 | 3/3 | 0/3 |
| prefix sum, OpenMP | 3/3 | 3/3 | 3/3 | 0/3 |
| prefix sum, MPI | 3/3 | 3/3 | 3/3 | 1/3 |
| histogram, OpenMP | 3/3 | 3/3 | 3/3 | 2/3 |
| histogram, MPI | 3/3 | 3/3 | 3/3 | 3/3 |
| **total (of 30)** | **30** | **30** | **28** | **12** |

Every entry matches the run records. The cell "prefix sum, MPI" for llama counts one pass
(the plain agent). My project log says that a later uniform re-grade of prefix sum
and histogram under MPI, for every model and at the same capped memory settings, did not
confirm that pass. That re-grade is not in the run records, so it is not applied here.

The prefix sum and histogram runs used the older benchmark without a warm-up run. Their
pass and fail marks for correctness stand, but their speed figures should be re-checked
(see `docs/DESIGN.md`).

## 3. The two recall-resistant problems

Problems: `decay` (a running value in which each step takes the largest of three
quantities) and `reservoir` (a level clamped between empty and a capacity). OpenMP and MPI
only. Source: my project log, 6 July 2026. **Unverified: the run files for these were not kept,
so these numbers are quoted from the log and cannot be checked against records.**

| problem and backend | sonnet | kimi | gemma | llama |
|---|---|---|---|---|
| decay, OpenMP | 3/3 | 2/3 | 3/3 | 0/3 |
| decay, MPI | 3/3 | 2/3 | 3/3 | 0/3 |
| reservoir, OpenMP | 3/3 | 2/3 | 3/3 | 0/3 |
| reservoir, MPI | 3/3 | 2/3 | 2/3 | 0/3 |
| **total (of 12)** | **12** | **8** | **11** | **0** |

Per problem, over both backends (of 6): decay sonnet 6, kimi 4, gemma 6, llama 0;
reservoir sonnet 6, kimi 4, gemma 5, llama 0. Llama's programs compiled but computed wrong
answers (a checksum mismatch). Kimi's four failures were correct with one worker and wrong
at the boundaries between blocks with several.

The `decay` runs were made with a header that named the technique family, which was
reworded afterwards (see `docs/DESIGN.md`); the runs were not repeated.

## 4. Totals by model, backend and problem set

Passing runs, out of three agent designs per cell. Derived from sections 1 and 2.

| model | OpenMP (of 21) | MPI (of 21) | CUDA (of 15) | seven well-known problems (of 57) | first three (of 27) | four harder (of 30) |
|---|---|---|---|---|---|---|
| sonnet | 21 | 21 | 15 | 57 | 27 | 30 |
| kimi | 21 | 20 | 15 | 56 | 26 | 30 |
| gemma | 21 | 17 | 15 | 53 | 25 | 28 |
| llama | 12 | 8 | 6 | 26 | 14 | 12 |

The chart of the same data is `docs/figures/pass-rates.svg`.

## 5. Results by agent design

### 5.1 First three problems, all backends (36 runs per design)

Source: `results/manifest-july-2026.tsv`.

| design | all models | strong models (sonnet, kimi) | weak models (gemma, llama) |
|---|---|---|---|
| basic | 31/36 | 17/18 | 14/18 |
| pydantic | 30/36 | 18/18 | 12/18 |
| langgraph | 31/36 | 18/18 | 13/18 |

### 5.2 Four harder problems (40 runs per design)

Source: `results/manifest-july-2026.tsv`, recomputed without the re-grade.

| design | all models | strong models | weak models |
|---|---|---|---|
| basic | 36/40 | 20/20 | 16/20 |
| pydantic | 32/40 | 20/20 | 12/20 |
| langgraph | 32/40 | 20/20 | 12/20 |

### 5.3 Weak models over all seven problems (38 runs per design)

Sum of 5.1 and 5.2 for gemma and llama.

| design | passes (of 38) |
|---|---|
| basic | 30 |
| langgraph | 25 |
| pydantic | 24 |

My project log, which applies the re-grade of Llama's prefix sum under MPI, gives 29 of 38
for the plain agent, and Llama's total for the four harder problems is 11 of 30 rather than
12. The ordering of the three designs is unchanged.

The plain agent led in both problem sets taken separately, and the LangGraph agent was
level with or ahead of the Pydantic agent in both. By backend, for the weak
models, over the seven problems: OpenMP basic 12, langgraph 11, pydantic 10; MPI basic 10,
langgraph 9, pydantic 6; CUDA basic 8, langgraph 5, pydantic 8.

### 5.4 By design and model, seven problems (19 runs per model and design)

| model | basic | langgraph | pydantic |
|---|---|---|---|
| sonnet | 19 | 19 | 19 |
| kimi | 18 | 19 | 19 |
| gemma | 18 | 18 | 17 |
| llama | 12 | 7 | 7 |

### 5.5 Recall-resistant problems

Source: my project log (records not kept). Passes out of 16 runs per design across the
four models: pydantic 11, basic 10, langgraph 10. The difference sits in one model. For kimi,
pydantic passed 4 of 4 and basic and langgraph 2 of 4 each. Sonnet passed with every design
and llama failed with every design. Gemma's one failure was its pydantic run under MPI on
`reservoir`.

### 5.6 Failure by budget exhaustion

Four runs in the whole study ended because the request budget was used up. Three were
Pydantic runs, including Llama on integration under MPI and Gemma on the heat stencil under
MPI; the log line is `The next request would exceed the request_limit of 200`. This count is
from my notes rather than from the manifest, which records only pass or fail.

## 6. Token use and cost

### 6.1 Four harder problems

Source: `results/tokens-new-problems-july-2026.json`; ten combinations of problem and
backend, three agent designs each, 5 and 6 July 2026. Totals: 3,322 requests, 51,827,746
prompt tokens and 628,083 completion tokens. Prices are the gateway's list prices per
million tokens: sonnet 3 dollars for prompt and 15 for completion tokens; kimi 0.60 and
3.03; llama 0.72 for both. The costs are computed here from the counts and these prices. Llama's 39.6 million prompt
tokens and 0.36 million completion tokens make the 39.9 million quoted in the README.

| model | requests | prompt tokens | completion tokens | cost | per combination (of 10) |
|---|---|---|---|---|---|
| kimi | 260 | 1,053,500 | 66,790 | $0.83 | about $0.08 |
| sonnet | 355 | 2,358,784 | 114,973 | $8.80 | about $0.88 |
| llama | 1,944 | 39,574,086 | 358,291 | $28.75 | about $2.88 |
| gemma | 763 | 8,841,376 | 88,029 | about $5 to $6 | about $0.50 to $0.60 |

Gemma was billed by the hour of a rented graphics-card instance at about 2 dollars an
hour, not by the token, so its cost is an estimate of instance time. The project log
quotes Llama's cost per combination as about $2.60. These counts give $2.88 and I could
not reproduce the log's figure.

### 6.2 Recall-resistant problems

Source: my project log (token snapshots not kept). Prompt and completion tokens in
millions, and cost at the same prices.

| model | prompt tokens | completion tokens | cost | passes |
|---|---|---|---|---|
| kimi | 9.1 million | 0.25 million | about $6.2 | 8 of 12 |
| sonnet | 5.0 million | 0.19 million | about $17.9 | 12 of 12 |
| llama | 32.5 million | 0.11 million | about $23.5 | 0 of 12 |
| gemma | not recorded | not recorded | about $1.2 (35 minutes of instance time) | 11 of 12 |

Total for the campaign: about $49. Every model needed more iteration than on the
well-known problems. In both campaigns the weakest model was by far the most expensive.

## 7. Speed-ups of the known-good reference solutions

These show that the problems can be parallelised and that the speed-up floors are reachable.
They are my measurements from the project log on a machine with 22 processor cores (not in
the manifests), one run each. The repeat count, the timer and the processor model are not recorded in this
repository, so these figures cannot be reproduced from it without re-measuring. The method
now in the code is wall-clock time (`omp_get_wtime`, `MPI_Wtime` or `clock_gettime`,
depending on the backend) in each problem's `bench.c`, with the speed-up taken against the
one-worker run. The prefix sum and histogram figures used the older
benchmark form.

| problem | OpenMP | MPI | CUDA |
|---|---|---|---|
| N-body | 7.2 | 4.1 | 7.0 |
| conjugate gradient | 4.8 | 3.9 | 15.4 |
| prefix sum | 5.9 | 4.0 | none |
| histogram | 4.5 | 3.4 | none |
| decay | 8.8 | 5.3 | none |
| reservoir | 7.8 | 4.2 | none |
| matrix multiply | not recorded here | 2.0 to 3.6 | 12.7 |
| heat stencil | not recorded here | 3.5 | 7.6 |
| integration | 6.8 | 4.3 | 4.5 |

A serial kernel measured about 1.0 on the benchmarks with the warm-up run and failed the
floor, as intended.

A later run on a 12-core host with a 16 GB NVIDIA graphics card passed every reference solution on OpenMP, MPI and CUDA, and a masked MPI reference; the speed-ups are in [results/reference-gates-gpu-host.md](../results/reference-gates-gpu-host.md).

## 8. Calling a finished kernel from Python

Demonstrations in `bindings/`, using nanobind wrappers around passing kernels. Source: my
project log; one run each. The notebook loop is the original serial Python.

| kernel | notebook loop | wrapped C | note |
|---|---|---|---|
| integration, OpenMP | 1.33 s | 0.004 s | 365 times faster; answer matches the oracle |
| matrix multiply, OpenMP | 1.49 s | 0.012 s | 123 times faster; the optimised numpy product takes 0.004 s |
| heat stencil, OpenMP | not recorded | not recorded | 31 times faster (unverified: the timings were not recorded, and `bindings/README.md` does not repeat it) |
| integration, MPI | 0.0152 s on one process | 0.0044 s on four | 3.4 times faster; answers differ by 6e-14, inside the measured spread |
| integration, CUDA | not recorded | 2.3 ms | matches the oracle |
