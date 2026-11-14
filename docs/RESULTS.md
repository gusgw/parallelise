# Results

This document reports how many of the routines that agents wrote in July and October 2026 are
correct, by the checks then in use and by the judgement that [DESIGN.md](DESIGN.md) describes,
applied again on 7 November 2026, what the reverse conversion found when it read them back into
plain Python, and how fast the known-good routines are. Each table names the files it comes from,
all under `results/`. Each combination was judged from a single run, so a
difference of a few runs may be chance.

## The studies

In July, three agent designs (basic, pydantic and langgraph) and four models converted seven of the
problems that DESIGN.md describes: integrate, matmul, jacobi, nbody and cg (conjugate gradients)
for OpenMP (threads within one computer), MPI (the Message Passing Interface, processes that
exchange messages) and CUDA (NVIDIA's platform for computing on graphics cards), and scan and
histogram for OpenMP and MPI. They made 237 runs of 228 combinations of problem, backend, model and
design; where a combination was run more than once, its latest run counts, and 225 of the 228 left
a routine, for three Llama runs of nbody for OpenMP wrote none. In October the same designs and
four models converted integrate, jacobi, scan and cg for OpenMP and MPI, each problem once as
published and once masked: 192 runs. The October routines were also re-checked in early November
with the check of that time.

## How the routines were judged again

`scripts/rejudge.py` judged every stored routine on a workstation with six processor cores and an
NVIDIA graphics card, with six workers for OpenMP and MPI. The routines were written against
headers that no longer exist, so the script changed a copy of each, and none of its logic, as its
opening comment sets out. Among other changes, the header of expected values is replaced by the old
tolerance alone, so that a routine which reads an expected value no longer builds, and the MPI
routines of scan, which the old contract let leave each block of the result on its own process,
gain a gather to process 0. The serial routines that referee this judgement were translated by
`kimi-k2.5`, one of the models judged, but each had first agreed with the Python at every test size
and at the benchmark size. Another program used several of the host's processors meanwhile (the
load average reached 18), so only this judgement's verdicts of correctness are reported, not its
speed-ups. It used integrate's tolerance of that time, 1e-8; on 7 November the tolerance was
tightened to 1e-11, about twenty times the largest difference that honest orders of addition make
at the benchmark size.

## October

| Problem and backend | Runs | Passed in October | Passed the re-check | Correct now |
| --- | ---: | ---: | ---: | ---: |
| cg, MPI | 24 | 16 | 16 | 8 |
| cg, OpenMP | 24 | 19 | 18 | 18 |
| integrate, MPI | 24 | 24 | 24 | 24 |
| integrate, OpenMP | 24 | 24 | 24 | 21 |
| jacobi, MPI | 24 | 12 | 14 | 7 |
| jacobi, OpenMP | 24 | 20 | 20 | 14 |
| scan, MPI | 24 | 15 | 18 | 17 |
| scan, OpenMP | 24 | 16 | 18 | 18 |
| All | 192 | 146 | 152 | 127 |

Sources: `campaign-2026-10/runs.jsonl`, `campaign-2026-10/recheck-2026-11.jsonl`,
`rejudge-2026-11.jsonl`. The masked runs are correct now in 62 of 96 cases and the unmasked in 65
of 96; in October 70 and 76 of them passed.

## July

| Problem and backend | Routines | Passed in July | Correct now |
| --- | ---: | ---: | ---: |
| cg, CUDA | 12 | 9 | 9 |
| cg, MPI | 12 | 8 | 2 |
| cg, OpenMP | 12 | 11 | 9 |
| histogram, MPI | 12 | 12 | 12 |
| histogram, OpenMP | 12 | 11 | 9 |
| integrate, CUDA | 12 | 9 | 11 |
| integrate, MPI | 12 | 11 | 11 |
| integrate, OpenMP | 12 | 12 | 11 |
| jacobi, CUDA | 12 | 11 | 9 |
| jacobi, MPI | 12 | 6 | 4 |
| jacobi, OpenMP | 12 | 11 | 7 |
| matmul, CUDA | 12 | 11 | 11 |
| matmul, MPI | 12 | 9 | 9 |
| matmul, OpenMP | 12 | 12 | 12 |
| nbody, CUDA | 12 | 11 | 8 |
| nbody, MPI | 12 | 10 | 9 |
| nbody, OpenMP | 9 | 9 | 9 |
| scan, MPI | 12 | 10 | 9 |
| scan, OpenMP | 12 | 9 | 9 |
| All | 225 | 192 | 170 |

Sources: `runs.jsonl`, `rejudge-2026-11.jsonl`. Five Llama routines (two of cg for MPI, one of cg
for OpenMP, one of integrate for OpenMP and one of nbody for CUDA) compiled in July but no longer
compile, because they define the routine with other argument types, or in another order, than the
present header. Every July routine of the other three models follows the present prototype, but the
July headers are not kept, so the three of these five that passed in July have a row of their own
in the table of causes. Leaving all five out, 170 of 220 routines are correct now.

## Each model

| Study | Model | Routines | Passed then | Correct now |
| --- | --- | ---: | ---: | ---: |
| July | claude-sonnet-4.6 | 57 | 57 | 51 |
| July | gemma-31b | 57 | 53 | 53 |
| July | kimi-k2.5 | 57 | 56 | 47 |
| July | llama-3.3-70b | 54 | 26 | 19 |
| October | claude-sonnet-4.5 | 48 | 45 | 42 |
| October | gpt-oss-120b | 48 | 41 | 37 |
| October | kimi-k2.5 | 48 | 44 | 37 |
| October | llama-3.3-70b | 48 | 16 | 11 |

Sources: as for the two tables above.

## What changed, and why

Of the 146 October passes, 119 are still correct, 22 now fail at a test size, 4 hang (a call took
more than two minutes) and 1 no longer builds; 8 October failures are now correct. Of the 192 July
passes, 167 are still correct, 19 fail at a test size, 3 hang and 3 no longer build; 3 July
failures are now correct. Most of the old passes that now fail meet inputs that the old checks
never used: their tests ran one size per problem, such as a grid of side 64 for cg and 128 for
jacobi with 200 sweeps, with one MPI process and with four, as the files of commit 629768d show.
Their causes are these:

| Cause | October | July |
| --- | ---: | ---: |
| Wrong at size 1 | 8 | 5 |
| Wrong or hanging with six MPI processes on a small input (size 8 or less) | 8 | 6 |
| jacobi for OpenMP frees the caller's grid after an odd number of sweeps | 5 | 4 |
| jacobi for MPI stops with an error | 5 | 2 |
| nbody for CUDA is wrong at size 300 | 0 | 3 |
| jacobi for CUDA stops with an error | 0 | 1 |
| scan for MPI writes its block of the result in the wrong place | 0 | 1 |
| The routine reads an expected value, which no longer exists | 1 | 0 |
| The routine's definition differs from the header | 0 | 3 |
| All | 27 | 25 |

Sources: `rejudge-2026-11.jsonl` (the field `why`) and the stored routines. The October routine
that reads an expected value is the prefix sum that wrote zeros and the expected checksum. Of the
old failures now correct, nine (seven in October, two in July) had failed at the old benchmark
stage or at the agent's own test; one October routine had failed because its agent changed a
trusted file, and one July routine had failed its own test with several workers.

## The reverse conversion of the stored routines

In November the reverse conversion that DESIGN.md describes was applied to the stored routines, in
the form that the re-judgement had given them to fit today's headers. Kimi K2.5, at temperature
zero and with at most 20 requests for each routine, translated each back into plain Python on the
same workstation and in the same containers as the re-judgement, and the fixed rule read every
translation. It read every routine that is correct now and 60 of those that fail. The other failures
were left out, because their translations cost the most and their flags would mostly repeat what
the judgement had found, and two of them never build and cannot be translated.

| Study | Correct now | Of them flagged | Failing now, read | Of them flagged |
| --- | ---: | ---: | ---: | ---: |
| October | 127 | 4 | 45 | 7 |
| July | 170 | 8 | 15 | 3 |
| All | 297 | 12 | 60 | 10 |

Sources: `rejudge-2026-11.jsonl` and `reverse-2026-11.jsonl`.

Every translation of a routine that is correct now agreed with its C on fresh inputs. Of the 12
correct routines that were flagged, 11 compare the number of workers with 0 or 1, or test whether
a size divides by it, as in `nthreads > 1` before a serial path or `num_threads < 1` as a guard;
the twelfth, a CUDA routine for nbody, is flagged for an empty loop that its translator wrote and
its C does not have. None of the correct routines is wrong at a size where its code changes, so
none hides a shortcut above some size. Of the 60 failing routines that were read, six are wrong
where their code changes, which the judgement had already found; two compare the number of workers
with a fixed number, one writes to an argument that it may only read, and the translation of one
defines no function `routine`. The fixed model, asked for its opinion of each flagged routine,
answered "No" for all 22, so its opinion says nothing about them. The readings took 2,114
requests, about 7 US dollars at the October list prices.

## Speed of the known-good routines

No speed-up measured before 7 November is reported, and none should be relied on. The earlier
checks timed each routine against its own run with one worker, as the `bench.csv` files stored from
July show, and a routine can slow that run on purpose. Until 7 November, too, the judgement let the
idle processes of an MPI routine keep processors busy while the serial routine was timed, which
made MPI speed-ups too large, and it could note the start of a call late, which made any routine
that filled every processor look too fast; both faults are now mended.

The known-good routines, which exist to test the harness, were judged again on 8 November on the
same workstation, with nothing else of note running; all 39 are correct, integrate's at its new
tolerance. Their speed-ups over each problem's serial routine are reference figures for the
harness, not findings about models. The serial matrix product reads its second matrix by columns;
the known-good OpenMP routine, which does not, is 19 times as fast even on one thread, and this is
why matmul shows 70 times on OpenMP and 553 times on CUDA. DESIGN.md names the other slow serial
routines. The serial routine of reservoir takes 0.07, 0.10 or 0.20 seconds according to its input,
so whether a routine beats it on all five inputs depends on the inputs drawn: the known-good MPI and
CUDA routines lost on the inputs on which it was quickest. Each speed-up is the median, over the
five timed inputs, of the serial routine's time divided by the routine's, and an asterisk marks one
that the judgement does not count as faster, because the routine lost on at least one input.

| Problem | OpenMP | MPI | CUDA |
| --- | ---: | ---: | ---: |
| cg | 4.52 | 8.18 | 29.19 |
| compact | 4.10 | 3.64 | 2.03* |
| decay | 1.26 | 0.89* | 1.12 |
| histogram | 3.01 | 3.01 | 0.90* |
| integrate | 4.31 | 2.93 | 8.56 |
| ising | 9.79 | 5.73 | 59.74 |
| jacobi | 5.69 | 3.25 | 20.34 |
| matmul | 70.28 | 5.58 | 553.05 |
| md | 5.66 | 8.91 | 2.60 |
| nbody | 5.78 | 5.33 | 14.61 |
| reservoir | 4.15 | 1.38* | 1.42* |
| scan | 0.81* | 0.40* | 0.38* |
| sparse_cg | 6.39 | 2.21 | 7.87 |

Source: `known-good-2026-11.json`.

## Files

The stored routines are under `results/runs/` (July) and `results/campaign-2026-10/runs/`
(October), beside the agents' own tests. The October records also hold each run's requests and
tokens, and `prices.json` beside them gives the list prices per million tokens.
`reverse-2026-11.jsonl` holds the reverse conversion's reading of each routine that it read: how
its translation compared with its C, the rule's flags, the sizes at which it was checked again,
the model's use and, where something was flagged, the fixed model's opinion. The older evidence
that no kept document uses (the July manifest and token counts, the October run logs, the reference
measurements of October and early November, the old charts and the old design documents) remains in
commit 629768d.
