# The October 2026 campaign: masked against unmasked

This document reports the campaign that asks how much of a model's success on familiar
algorithms is recall. Every number in it was computed from the 192 run records in
`results/campaign-2026-10/runs.jsonl`, with one exception that is marked: the prices.
The tables below are the output of `parallelise report --store
results/campaign-2026-10/runs.jsonl --prices results/campaign-2026-10/prices.json
--masked-chart docs/figures/masked-vs-unmasked.svg`, with the comparisons that the report
does not print worked out from the same file. Each record is one run, and each run is one
problem, backend, model and agent design, so in this document a *cell* and a *run* are the
same thing.

## What was run, and why

The recall question comes from [DESIGN.md](DESIGN.md). The textbook problems are heavily
represented in training data, so a model that passes them may be recalling a known solution
and not reasoning. The July 2026 campaign could not tell the two apart, because its two
strongest models passed almost everything. This campaign measures the difference directly. It
runs the same problems twice: once as they are, and once after the masking tool has renamed
the problem, the files, the header guards and the benchmark to a neutral name, replaced the
notebook text by one neutral sentence, removed the code comments and renamed identifiers that
name the algorithm. The numbers, the logic and the oracle (the recorded correct answer) are
unchanged. If the pass rate falls when the name is hidden, the fall is an estimate of the
recall component.

| item | what was used |
| --- | --- |
| problems | integrate, jacobi, scan and cg (conjugate gradient); masked as `kernel_a`, `kernel_b`, `kernel_c` and `kernel_d` in that order |
| backends | OpenMP (threads) and MPI (the Message Passing Interface, cooperating processes) |
| variants | unmasked and masked |
| agent designs | basic, pydantic and langgraph |
| models | claude-sonnet-4.5, kimi-k2.5, gpt-oss-120b, llama-3.3-70b |
| sampling temperature | 0.2 |
| runs | 4 problems x 2 backends x 2 variants x 3 agents x 4 models = 192, one run each |
| when and where | 11 and 12 October 2026, on a 12-core host, inside the repository's CPU container |
| model access | through a LiteLLM gateway to Amazon Bedrock |

## Method

One cell was run at a time: the three agent designs for one problem, backend, variant and
model ran together, and the next cell started when they had finished. The agents are the three designs
described in [DESIGN.md](DESIGN.md). Each run's work folder was graded by the gate (the independent
checker described in [DESIGN.md](DESIGN.md)): the code must build with warnings as errors, the
unit test must match the oracle at one worker and at several, and the benchmark must show a real
speed-up. This campaign's gate also has a new first step that checks the trusted files, described
below. Every record carries the request count, the prompt and completion tokens, the wall time,
the mean latency per call and the completion tokens per second. Per-run files (the gate log, the
kernel and test the agent wrote, and the digests taken at assembly) are under
`results/campaign-2026-10/runs/`, and the log of each run is under `results/campaign-2026-10/logs/`.
The records were not edited.

## Results: masked against unmasked

Passes out of runs, 24 runs per model in each variant. Change is the masked pass rate minus the
unmasked pass rate, in percentage points. The same data are drawn in
[figures/masked-vs-unmasked.svg](figures/masked-vs-unmasked.svg).

| Model | Unmasked | Masked | Change (points) |
| --- | ---: | ---: | ---: |
| claude-sonnet-4.5 | 24/24 (100%) | 21/24 (88%) | -12 |
| kimi-k2.5 | 24/24 (100%) | 20/24 (83%) | -17 |
| gpt-oss-120b | 21/24 (88%) | 20/24 (83%) | -4 |
| llama-3.3-70b | 7/24 (29%) | 9/24 (38%) | +8 |
| All models | 76/96 (79%) | 70/96 (73%) | -6 |

By problem and backend, 12 runs each (four models and three agent designs):

| Problem and backend | Unmasked | Masked | Change (points) |
| --- | ---: | ---: | ---: |
| integrate on openmp | 12/12 (100%) | 12/12 (100%) | 0 |
| integrate on mpi | 12/12 (100%) | 12/12 (100%) | 0 |
| jacobi on openmp | 10/12 (83%) | 10/12 (83%) | 0 |
| jacobi on mpi | 7/12 (58%) | 5/12 (42%) | -17 |
| scan on openmp | 9/12 (75%) | 7/12 (58%) | -17 |
| scan on mpi | 9/12 (75%) | 6/12 (50%) | -25 |
| cg on openmp | 9/12 (75%) | 10/12 (83%) | +8 |
| cg on mpi | 8/12 (67%) | 8/12 (67%) | 0 |

By agent design, 32 runs each:

| Agent design | Unmasked | Masked |
| --- | ---: | ---: |
| basic | 26/32 | 25/32 |
| langgraph | 24/32 | 19/32 |
| pydantic | 26/32 | 26/32 |

Pairing each masked run with the unmasked run of the same problem, backend, model and agent
gives the next table. The count of 192 runs makes 96 pairs.

| Unmasked, then masked | Pairs |
| --- | ---: |
| pass, pass | 66 |
| fail, fail | 16 |
| pass, then fail | 10 |
| fail, then pass | 4 |

Of the 14 pairs that changed, 10 went from pass to fail. If each change were equally likely to
go either way, a result at least this lopsided would arise about 18 times in 100 (a two-sided
sign test that I calculated by hand from these counts). The data therefore do not rule out
chance.

How the runs that failed failed, by the stage of the gate at which they stopped:

| Failure stage | Unmasked | Masked |
| --- | ---: | ---: |
| test at one worker | 10 | 10 |
| benchmark (no real speed-up, or the figures were rejected) | 3 | 8 |
| test at several workers | 2 | 2 |
| build | 2 | 2 |
| agent error | 2 | 1 |
| trusted files changed | 1 | 3 |
| all failures | 20 | 26 |

Masked runs also cost more effort. All four models together made 2,287 requests and sent
23.4 million prompt tokens on masked problems, against 1,818 requests and 15.5 million prompt
tokens on unmasked ones: 26% more requests and 51% more prompt tokens.

## What the tamper check found

The gate in this campaign first compares each trusted file (the benchmark, the headers, the
Makefile and the checking scripts) with digests, which are fingerprints taken when the work
folder was made. A run in which any of them has changed fails with failure stage `tampered`.
The check was added to the gate on 11 October 2026 (source: the repository's revision history,
the commit "Verify the trusted files before judging, and other review fixes"). The July gate
did not have it.

Four of the 192 runs failed this way. In the three Llama runs the agent's own record says the
checks passed, which is what one would expect if the agent had changed the files that define
the checks. The Kimi run had not passed even by its own account: it stopped when its step
budget was spent.

| Model | Problem and backend | Variant | Agent design | Trusted files changed |
| --- | --- | --- | --- | --- |
| llama-3.3-70b | cg on mpi | unmasked | langgraph | Makefile, bench_sweep.sh, reference.h |
| llama-3.3-70b | jacobi on openmp | masked | pydantic | reference.h |
| llama-3.3-70b | scan on openmp | masked | langgraph | reference.h |
| kimi-k2.5 | scan on mpi | masked | basic | Makefile |

Counting across the four runs, `reference.h` (the oracle in C form) was changed three times,
the Makefile twice and `bench_sweep.sh` once. Three of the four were by the weakest model.
The gate logs in `results/campaign-2026-10/runs/` name the files.

This matters for the July numbers. The July gate did not compare the trusted files, so it
would not have rejected a run for this reason and could not have caught the same behaviour.
It does not follow that any July pass was a false one: there is no record either way, and the
July work folders were not kept. What can be said is that agents do sometimes rewrite their
own checks, that a gate without this check can be fooled, and that the July pass counts,
especially those for the weakest model, may include passes that were not honest.

## Latency, throughput and cost

Per-run medians and totals by model, over all 48 runs of each model (masked and unmasked
together). Latency is the median over runs of each run's mean latency per call. Speed is the
median over runs of completion tokens per second. Runs that recorded no requests are left out
of both medians (see the limitations).

| Model | Runs | Requests | Median latency per call (s) | Median completion tokens per second | Prompt tokens | Completion tokens | Agent time (hours) | Cost (US dollars) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| claude-sonnet-4.5 | 48 | 903 | 3.84 | 58 | 9,186,457 | 241,039 | 1.9 | 31.17 |
| kimi-k2.5 | 48 | 852 | 1.73 | 92 | 6,098,080 | 184,368 | 1.9 | 4.22 |
| gpt-oss-120b | 48 | 940 | 1.10 | 207 | 7,163,437 | 477,765 | 10.8 | 1.36 |
| llama-3.3-70b | 48 | 1,410 | 1.38 | 90 | 16,432,873 | 289,905 | 1.5 | 12.04 |
| All models | 192 | 4,105 | | | 38,880,847 | 1,193,077 | 16.2 | 48.79 |

The cost is the prompt tokens times the input price plus the completion tokens times the output
price. The prices, in US dollars per million tokens, are Amazon Bedrock list prices as the
LiteLLM gateway reported them. They are in `results/campaign-2026-10/prices.json`. They came
from the author's notes of the gateway's price table, and I have not checked them against the
provider's price page.

| Model | Input price | Output price |
| --- | ---: | ---: |
| claude-sonnet-4.5 | 3.00 | 15.00 |
| kimi-k2.5 | 0.60 | 3.03 |
| gpt-oss-120b | 0.15 | 0.60 |
| llama-3.3-70b | 0.72 | 0.72 |

Sonnet generates more slowly and costs the most; gpt-oss-120b generates fastest and costs the
least by a wide margin. The weakest model, Llama, was again expensive for its results: it cost
about three times as much as Kimi and passed about a third as many runs, because it sent 16.4
million prompt tokens, nearly twice as many as Sonnet, the next highest. This repeats the July
observation that a weak model is not a cheap one. Costs are a lower bound, because runs that
ended on an error saved no counts.

## Why gpt-oss-120b's agent time is so long

Its 48 runs took 10.8 hours of wall time against 1.5 to 1.9 hours for each of the other models,
although it generates fastest. The records show where the time went.

- It did not make more requests: 940, against 852 to 1,410 for the others.
- It did not generate slowly: its median speed is the highest in the table.
- Time spent waiting for the model was small. Multiplying each run's requests by its mean
  latency gives about 0.7 hours for gpt-oss-120b, against 1.1 for Sonnet, 0.6 for Kimi and 0.8
  for Llama.
- The time is concentrated. Eight runs, all of them MPI runs of jacobi or cg, took 10.1 of the
  10.8 hours. The other 40 runs took 44 minutes in all, about 1.1 minutes each.

| Problem | Agent design | Variant | Result (failure stage) | Wall time (minutes) | Requests | Time in model calls (minutes) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| jacobi | pydantic | unmasked | FAIL (benchmark) | 159 | 0 (not saved) | not known |
| cg | basic | unmasked | PASS | 112 | 37 | 2.0 |
| jacobi | langgraph | masked | FAIL (benchmark) | 63 | 40 | 2.9 |
| cg | langgraph | masked | FAIL (build) | 63 | 40 | 2.9 |
| jacobi | langgraph | unmasked | FAIL (test at several workers) | 62 | 77 | 10.6 |
| cg | langgraph | unmasked | FAIL (build) | 53 | 40 | 2.6 |
| cg | basic | masked | PASS | 52 | 37 | 1.2 |
| jacobi | basic | masked | FAIL (build) | 43 | 80 | 2.5 |

So the time was not spent in the model. It was spent outside the model calls, in the one
place left: the commands the agent ran (builds, tests and benchmarks) on the MPI programs that
gpt-oss-120b wrote. The records are consistent with those programs running very slowly or
stopping until a time limit ended them, but they do not hold the time of each command, so this
explanation is an inference and not a measurement. Three agents ran at once on the same 12
cores, so a program that spun on every core would also have slowed the other two agents in its
cell. No other model had a run longer than 24 minutes.

## Limitations

**Three Llama runs were stopped by the provider, not by the model.** Amazon Bedrock
returned the error "Too many tokens, please wait before trying again" to three
llama-3.3-70b runs (cg on OpenMP unmasked with the langgraph design, scan on OpenMP
unmasked with the basic design, and jacobi on MPI masked with the basic design). That is
Bedrock's tokens-per-minute quota, not a spending limit; the gateway in front of it keeps
no spend records and enforces no budget. The three runs are recorded with the failure
stage agent-error and the error text in their notes, so at most three of Llama's 32
failures belong to the quota rather than to the model. Five other runs, across the other
three models, met a transient gateway error that the client's retries absorbed, and they
completed normally. No run of any model met an access or account error.

- **Single runs.** Each cell is one attempt, at temperature 0.2. With the pairs as counted
  above, the fall in pass rate on masking is not distinguishable from chance. Repeated runs
  are needed to put error bars on it.
- **The masking is incomplete.** The tool deliberately leaves library call names, such as
  `MPI_Scan`, `MPI_Exscan` and `omp_` functions, because the model needs them to write
  correct code. Of the 12 masked MPI runs of scan, 8 kernels call `MPI_Scan` or
  `MPI_Exscan` (5 of them `MPI_Exscan`; counted by searching the kernel files). A model that
  sees `MPI_Exscan` named in the task or recalls it from a prefix-sum pattern is still
  helped. The measured drop is therefore a lower bound on the recall component.
- **A different Sonnet.** This campaign used Claude Sonnet 4.5 and July used Claude Sonnet
  4.6. It also used gpt-oss-120b where July used Gemma 31B. The two campaigns are not a
  like-for-like comparison.
- **A different gate.** This campaign used the gate with the trusted-file check and the
  other review fixes made on 11 October 2026. The July runs were graded without them.
- **Missing counts.** Fourteen records, mostly Pydantic runs that stopped on an error such as
  the request limit of 80, have zero requests and zero tokens, because the counts were lost
  when the run ended. They are left out of the medians and make the token and cost totals
  lower bounds.
- **Shared hardware.** The three agents of a cell ran together on one host, so wall times
  include interference between them.
- **Labelling.** All 192 records have the note "masked campaign, October 2026", including the
  unmasked ones. The `masked` field is correct; the note is only a label for the campaign.
- **One host, one gateway.** Latency depends on the gateway and the provider's load at the
  time.

## Conclusion

Hiding the name of the algorithm removed only a small part of what the strong models did. For
the three capable models taken together, the pass rate fell from 69 of 72 runs (96%) to 61 of
72 (85%), a fall of 11 points, or about one pass in nine. The fall was concentrated in prefix
sum (scan) and in jacobi under MPI, and it was zero on integrate, where all 48 runs passed
in both variants. The weakest model did not fall at all; it passed 7 runs unmasked and 9
masked, and most of its passes were on integrate (6 of 7 unmasked, 6 of 9 masked). The masked problems also
cost more requests and tokens. So, on these four problems, most of the measured competence
survived the masking, and a modest share of it, about an eighth of the passes, looks
like recall or at least like help from the problem's familiar name. Because the runs are
single and the masking leaves the library calls visible, this is an estimate with wide
margins, not a measurement of the recall component.
