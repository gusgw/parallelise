# Results

What the harness has produced so far, as tables. Each table names the file it was computed
from. The experimental design is in [DESIGN.md](DESIGN.md); the October 2026 campaign has
its own document, [CAMPAIGN-2026-10.md](CAMPAIGN-2026-10.md).

How to read the tables. A run is one attempt by one agent design, with one model, on one
problem and backend; it passes or fails the check. The July study and the October campaign
were judged by an earlier rule under which a correct routine whose best speed-up was below
the problem's floor failed at the benchmark stage; under the present rule a pass means a
correct routine, whatever its speed, and speed is reported apart. Section 6 gives the October
counts under both rules. A cell is one problem and backend for
one model, scored out of the three agent designs, so "2/3" means two designs passed. Every
run is a single attempt; where a cell was run more than once, the latest record counts.
The backends are OpenMP (threads in one computer), MPI (cooperating processes) and CUDA
(a graphics processor). In column headings, sonnet is `claude-sonnet-4.6`, kimi is
`kimi-k2.5`, gemma is `gemma-31b` and llama is `llama-3.3-70b`, unless a table says
otherwise.

## 1. The first three problems

Integration by the midpoint rule, matrix multiply and a heat stencil (the Jacobi method);
three backends; four models. Source: `results/manifest-july-2026.tsv`, 4 and 5 July 2026,
108 runs. Each entry counts passes out of the three agent designs.

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

## 2. The four harder problems

N-body forces, conjugate gradient, prefix sum and histogram. Prefix sum and histogram have
no CUDA version. Source: `results/manifest-july-2026.tsv`, 5 July 2026, 120 runs.

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

## 3. The two recall-resistant problems

`decay` (a running value in which each step takes the largest of three quantities) and
`reservoir` (a level clamped between empty and a capacity); OpenMP and MPI. Source: the
project's working notes of 6 July 2026; the run files were not kept.

| problem and backend | sonnet | kimi | gemma | llama |
|---|---|---|---|---|
| decay, OpenMP | 3/3 | 2/3 | 3/3 | 0/3 |
| decay, MPI | 3/3 | 2/3 | 3/3 | 0/3 |
| reservoir, OpenMP | 3/3 | 2/3 | 3/3 | 0/3 |
| reservoir, MPI | 3/3 | 2/3 | 2/3 | 0/3 |
| **total (of 12)** | **12** | **8** | **11** | **0** |

## 4. Totals by model and backend

Passing runs over the seven problems of sections 1 and 2. The chart of the same data is
[figures/pass-rates.svg](figures/pass-rates.svg).

| model | OpenMP (of 21) | MPI (of 21) | CUDA (of 15) | all seven (of 57) | first three (of 27) | four harder (of 30) |
|---|---|---|---|---|---|---|
| sonnet | 21 | 21 | 15 | 57 | 27 | 30 |
| kimi | 21 | 20 | 15 | 56 | 26 | 30 |
| gemma | 21 | 17 | 15 | 53 | 25 | 28 |
| llama | 12 | 8 | 6 | 26 | 14 | 12 |

## 5. Results by agent design

Source: `results/manifest-july-2026.tsv`. "Strong models" are sonnet and kimi; "weak
models" are gemma and llama.

First three problems, all backends, 36 runs per design:

| design | all models | strong models | weak models |
|---|---|---|---|
| basic | 31/36 | 17/18 | 14/18 |
| pydantic | 30/36 | 18/18 | 12/18 |
| langgraph | 31/36 | 18/18 | 13/18 |

Four harder problems, 40 runs per design:

| design | all models | strong models | weak models |
|---|---|---|---|
| basic | 36/40 | 20/20 | 16/20 |
| pydantic | 32/40 | 20/20 | 12/20 |
| langgraph | 32/40 | 20/20 | 12/20 |

By design and model over the seven problems, 19 runs per entry:

| model | basic | langgraph | pydantic |
|---|---|---|---|
| sonnet | 19 | 19 | 19 |
| kimi | 18 | 19 | 19 |
| gemma | 18 | 18 | 17 |
| llama | 12 | 7 | 7 |

Recall-resistant problems, 16 runs per design across the four models (source: the working
notes of 6 July 2026): pydantic 11, basic 10, langgraph 10.

## 6. The October 2026 campaign: masked against unmasked

Integration, the heat stencil, the prefix sum and conjugate gradient on OpenMP and MPI, each
run as shipped and again after the masking tool had replaced every revealing name; four
models (here sonnet is `claude-sonnet-4.5`); three agent designs; one run per cell; 192
runs. Source: `results/campaign-2026-10/runs.jsonl`. Passes out of 24 runs per model and
variant. The chart is [figures/masked-vs-unmasked.svg](figures/masked-vs-unmasked.svg);
the full tables, with latency, throughput and cost, are in
[CAMPAIGN-2026-10.md](CAMPAIGN-2026-10.md).

| model | unmasked | masked |
|---|---|---|
| claude-sonnet-4.5 | 24/24 | 21/24 |
| kimi-k2.5 | 24/24 | 20/24 |
| gpt-oss-120b | 21/24 | 20/24 |
| llama-3.3-70b | 7/24 | 9/24 |
| all models | 76/96 | 70/96 |

The same runs counted under the present rule, in which a correct routine passes whatever its
speed. Source: `results/campaign-2026-10/runs.jsonl` and the check log `gate.log` of each run
that failed at the benchmark stage, under `results/campaign-2026-10/runs/`. Seven runs change
from a failure to a correct routine with no useful speed-up; the list is in
[CAMPAIGN-2026-10.md](CAMPAIGN-2026-10.md). Correct runs out of 24 per model and variant.

| model | earlier rule, unmasked | earlier rule, masked | present rule, unmasked | present rule, masked |
|---|---|---|---|---|
| claude-sonnet-4.5 | 24/24 | 21/24 | 24/24 | 24/24 |
| kimi-k2.5 | 24/24 | 20/24 | 24/24 | 21/24 |
| gpt-oss-120b | 21/24 | 20/24 | 22/24 | 22/24 |
| llama-3.3-70b | 7/24 | 9/24 | 7/24 | 9/24 |
| three capable models (of 72) | 69/72 | 61/72 | 70/72 | 67/72 |
| all models (of 96) | 76/96 | 70/96 | 77/96 | 76/96 |

Correct runs out of 12 per problem, backend and variant, under the present rule:

| problem and backend | unmasked | masked |
|---|---|---|
| integration, OpenMP | 12/12 | 12/12 |
| integration, MPI | 12/12 | 12/12 |
| heat stencil, OpenMP | 10/12 | 10/12 |
| heat stencil, MPI | 8/12 | 6/12 |
| prefix sum, OpenMP | 9/12 | 9/12 |
| prefix sum, MPI | 9/12 | 8/12 |
| conjugate gradient, OpenMP | 9/12 | 11/12 |
| conjugate gradient, MPI | 8/12 | 8/12 |

The same runs re-checked on 6 November 2026 with the present check: each work folder judged as
it was assembled, with the problem's own known-good test program in place of the agent's and,
for OpenMP, every answer at the benchmark's size compared with the known-good routine's.
Source: `results/campaign-2026-10/recheck-2026-11.jsonl`; the method and the 14 runs whose
verdict changed are in [CAMPAIGN-2026-10.md](CAMPAIGN-2026-10.md). Correct runs out of 24 per
model and variant:

| model | unmasked | masked |
|---|---|---|
| claude-sonnet-4.5 | 24/24 | 23/24 |
| kimi-k2.5 | 24/24 | 21/24 |
| gpt-oss-120b | 22/24 | 22/24 |
| llama-3.3-70b | 10/24 | 6/24 |
| three capable models (of 72) | 70/72 | 66/72 |
| all models (of 96) | 80/96 | 72/96 |

Correct runs out of 12 per problem, backend and variant, re-checked:

| problem and backend | unmasked | masked |
|---|---|---|
| integration, OpenMP | 12/12 | 12/12 |
| integration, MPI | 12/12 | 12/12 |
| heat stencil, OpenMP | 11/12 | 9/12 |
| heat stencil, MPI | 8/12 | 6/12 |
| prefix sum, OpenMP | 9/12 | 9/12 |
| prefix sum, MPI | 11/12 | 7/12 |
| conjugate gradient, OpenMP | 9/12 | 9/12 |
| conjugate gradient, MPI | 8/12 | 8/12 |

## 7. Token use and cost

Four harder problems, ten combinations of problem and backend, three agent designs each.
Source: `results/tokens-new-problems-july-2026.json`, 5 and 6 July 2026, counted by a
proxy between the agents and the gateway. Prices are list prices per million tokens:
sonnet 3 dollars for prompt and 15 for completion tokens, kimi 0.60 and 3.03, llama 0.72
for both; gemma ran on a rented graphics-card instance at about 2 dollars an hour.

| model | requests | prompt tokens | completion tokens | cost | per combination |
|---|---|---|---|---|---|
| kimi | 260 | 1,053,500 | 66,790 | $0.83 | about $0.08 |
| sonnet | 355 | 2,358,784 | 114,973 | $8.80 | about $0.88 |
| llama | 1,944 | 39,574,086 | 358,291 | $28.75 | about $2.88 |
| gemma | 763 | 8,841,376 | 88,029 | about $5 to $6 of instance time | about $0.50 to $0.60 |

Recall-resistant problems, from the working notes of 6 July 2026, at the same prices:

| model | prompt tokens | completion tokens | cost | passes |
|---|---|---|---|---|
| kimi | 9.1 million | 0.25 million | about $6.20 | 8 of 12 |
| sonnet | 5.0 million | 0.19 million | about $17.90 | 12 of 12 |
| llama | 32.5 million | 0.11 million | about $23.50 | 0 of 12 |
| gemma | not recorded | not recorded | about $1.20 of instance time | 11 of 12 |

The October 2026 campaign cost about $49 at Bedrock list prices: sonnet $31.17, llama
$12.04, kimi $4.22, gpt-oss $1.36. Its per-model request, token, latency and throughput
figures are in [CAMPAIGN-2026-10.md](CAMPAIGN-2026-10.md).

## 8. Speed-ups of the known-good reference solutions

Best speed-up of each reference solution over its one-worker run, one measurement each.
The first table is from the working notes of July 2026 on a machine with 22 processor
cores; the second is from 11 October 2026 on a 12-core host with a 16 GB NVIDIA graphics
card, recorded in [../results/reference-checks-gpu-host.md](../results/reference-checks-gpu-host.md)
with each problem's floor; the third is from 6 November 2026 on the same host, recorded in
`results/reference-speedups-2026-11.json`. Only the November figures were measured with the
untimed warm-up call in every OpenMP and MPI benchmark. In November the MPI benchmarks of the
prefix sum, histogram, decay and reservoir ran at a length of 20 million with at most 4
processes, to fit an 8 GB memory limit. In November the known-good prefix sum under MPI
reached 1.16, below its floor of 1.25; under the present rule it is a correct routine with no
useful speed-up.

| problem | OpenMP (July) | MPI (July) | CUDA (July) | OpenMP (October) | MPI (October) | CUDA (October) | OpenMP (November) | MPI (November) | CUDA (November) |
|---|---|---|---|---|---|---|---|---|---|
| integration | 6.8 | 4.3 | 4.5 | 5.30 | 3.92 | 8.95 | 5.45 | 5.86 | 8.95 |
| matrix multiply | not recorded | 2.0 to 3.6 | 12.7 | 5.05 | 8.95 | 27.53 | 5.14 | 6.42 | 25.62 |
| heat stencil | not recorded | 3.5 | 7.6 | 6.18 | 5.42 | 11.51 | 6.01 | 6.16 | 11.76 |
| N-body | 7.2 | 4.1 | 7.0 | 6.03 | 5.72 | 15.23 | 6.05 | 5.78 | 15.21 |
| conjugate gradient | 4.8 | 3.9 | 15.4 | 4.59 | 3.51 | 17.60 | 5.76 | 3.44 | 17.61 |
| prefix sum | 5.9 | 4.0 | none | 1.93 | 1.32 | none | 1.49 | 1.16 | none |
| histogram | 4.5 | 3.4 | none | 3.09 | 3.01 | none | 3.17 | 3.13 | none |
| decay | 8.8 | 5.3 | none | 6.58 | 7.05 | none | 6.51 | 3.84 | none |
| reservoir | 7.8 | 4.2 | none | 7.02 | 5.80 | none | 7.00 | 3.96 | none |

## 9. Calling a finished routine from Python

Demonstrations in `bindings/`, using nanobind wrappers around routines that passed the
check; one run each, from the working notes of July 2026. The notebook loop is the
original serial Python.

| routine | notebook loop | wrapped C | note |
|---|---|---|---|
| integration, OpenMP | 1.33 s | 0.004 s | 365 times faster; the answer matches the oracle |
| matrix multiply, OpenMP | 1.49 s | 0.012 s | 123 times faster; the optimised numpy product takes 0.004 s |
| heat stencil, OpenMP | not recorded | not recorded | 31 times faster |
| integration, MPI | 0.0152 s on one process | 0.0044 s on four | 3.4 times faster; the answers differ by 6e-14 |
| integration, CUDA | not recorded | 2.3 ms | the answer matches the oracle |

## 10. Recommended backends against measured speed-ups

The backend that each model recommended for each notebook, from `parallelise assess`, set
against the best speed-up of the known-good solution on each backend. In this section sonnet
is `claude-sonnet-4.5` and kimi is `kimi-k2.5`. For a combination, the backend shown in
brackets is the one the harness would run, the first part that is not serial. "No version"
means that the problem has no CUDA version. Recommendations: `results/assessments/2026-11/`,
one call per notebook and model, 6 November 2026. Speed-ups:
`results/reference-speedups-2026-11.json`, one measurement each, 6 November 2026, on a 12-core
host with a 16 GB NVIDIA graphics card, with the untimed warm-up call in every benchmark.
OpenMP used 1 to 12 threads at each benchmark's default length. MPI used 1 to 12 processes at
the default length, except for decay, histogram, reservoir and the prefix sum, whose MPI
benchmarks were run at a length of 20 million with at most 4 processes to fit an 8 GB memory
limit; for those four problems the MPI and OpenMP figures are not measured alike. The CUDA
speed-up is over one processor core. A backend is called the fastest only where it beats the
next by at least 5 per cent; closer than that, two single measurements cannot separate them.
Where the processor backends were not measured alike, a recommendation between them cannot be
judged.

| problem | OpenMP | MPI | CUDA | fastest | faster of OpenMP and MPI | sonnet recommends | sonnet fastest | kimi recommends | kimi fastest |
|---|---|---|---|---|---|---|---|---|---|
| integration | 5.45 | 5.86 | 8.95 | CUDA | MPI | OpenMP | no | OpenMP | no |
| matrix multiply | 5.14 | 6.42 | 25.62 | CUDA | MPI | OpenMP | no | combination (runs OpenMP) | no |
| heat stencil (Jacobi) | 6.01 | 6.16 | 11.76 | CUDA | about equal | OpenMP | no | OpenMP | no |
| N-body | 6.05 | 5.78 | 15.21 | CUDA | about equal | OpenMP | no | combination (runs OpenMP) | no |
| conjugate gradient | 5.76 | 3.44 | 17.61 | CUDA | OpenMP | OpenMP | no | OpenMP | no |
| prefix sum (scan) | 1.49 | 1.16 | no version | OpenMP or MPI | not measured alike | OpenMP | cannot be judged | CUDA | no (no version) |
| histogram | 3.17 | 3.13 | no version | OpenMP or MPI | not measured alike | OpenMP | cannot be judged | OpenMP | cannot be judged |
| decay | 6.51 | 3.84 | no version | OpenMP or MPI | not measured alike | serial | no | serial | no |
| reservoir | 7.00 | 3.96 | no version | OpenMP or MPI | not measured alike | serial | no | serial | no |