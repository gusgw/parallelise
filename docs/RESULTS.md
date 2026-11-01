# Results

What the harness has produced so far, as tables. Each table names the file it was computed
from. The experimental design is in [DESIGN.md](DESIGN.md); the October 2026 campaign has
its own document, [CAMPAIGN-2026-10.md](CAMPAIGN-2026-10.md).

How to read the tables. A run is one attempt by one agent design, with one model, on one
problem and backend; it passes or fails the check. A cell is one problem and backend for
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
with each problem's floor.

| problem | OpenMP (July) | MPI (July) | CUDA (July) | OpenMP (October) | MPI (October) | CUDA (October) |
|---|---|---|---|---|---|---|
| integration | 6.8 | 4.3 | 4.5 | 5.30 | 3.92 | 8.95 |
| matrix multiply | not recorded | 2.0 to 3.6 | 12.7 | 5.05 | 8.95 | 27.53 |
| heat stencil | not recorded | 3.5 | 7.6 | 6.18 | 5.42 | 11.51 |
| N-body | 7.2 | 4.1 | 7.0 | 6.03 | 5.72 | 15.23 |
| conjugate gradient | 4.8 | 3.9 | 15.4 | 4.59 | 3.51 | 17.60 |
| prefix sum | 5.9 | 4.0 | none | 1.93 | 1.32 | none |
| histogram | 4.5 | 3.4 | none | 3.09 | 3.01 | none |
| decay | 8.8 | 5.3 | none | 6.58 | 7.05 | none |
| reservoir | 7.8 | 4.2 | none | 7.02 | 5.80 | none |

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
