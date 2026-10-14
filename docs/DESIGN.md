# Experimental design

This document explains how the experiment is built and what it found, for a reader who
wants to judge whether the conclusions can be trusted. The code is mapped in
`docs/ARCHITECTURE.md`; every table of results is in `docs/RESULTS.md`, and each number
quoted here comes from the tables there. The runs described were made on 4 to 6 July 2026.

## The question

Can a language model turn a serial research notebook into a parallel C program that is
correct and genuinely faster, and what decides whether it succeeds? Three things are varied
on purpose, and everything else is held fixed.

## The three controlled axes

**Model.** Four models are reached through one OpenAI-compatible gateway with the same
prompts, the same tools and the same budgets: `claude-sonnet-4.6`, `kimi-k2.5`,
`llama-3.3-70b` and `gemma-31b`. They were chosen to span a range: a leading proprietary
model, a strong open-weight model, a large older open-weight model kept as the weak control
that separates the limits of a model from the limits of the harness, and a medium recent
open-weight model. Serving details are recorded as far as they are known (see the README).

**Agent design.** Three agents share one tool surface: read the notebook, read a file,
write a file, and run an allow-listed build command. They share one finishing rule: the work
is not done until the checks pass. They differ only in how that rule is enforced.

- `basic` is a loop written directly against the HTTP interface with the `httpx` library. The
  model works with the tools. When it stops calling tools, the loop runs the checks itself.
  If they pass, the run is done. If not, the loop appends the check output as a new message
  and lets the model continue, up to a budget of 80 steps. When the budget ends, the files
  are left on disk and are still graded.
- `langgraph` wraps the same idea in the LangGraph framework. The model runs to a natural
  stopping point, the surrounding code runs the checks, and on failure it appends the output
  and starts another full attempt, up to 8 attempts.
- `pydantic` uses the output validator of the Pydantic AI framework. A validator ordinarily
  checks that a model's answer has the right type and, if not, raises a retry so that the
  model tries again. Here the validator checks that the build and tests pass, and raises the
  same retry until they do or until the request limit is reached. The shipped default for this limit
  is the step budget of 80 (`--step-budget`); the July runs used a limit of 200, as the log
  line quoted below shows.

**Backend.** The same algorithm is converted for OpenMP (threads in one computer), MPI
(cooperating processes) and CUDA (a graphics processor). The prompts give the model a menu
of patterns and leave the choice to it: a reduction with `MPI_Allreduce`, gathering distinct
output slices with `MPI_Allgather` or `MPI_Allgatherv`, and a halo exchange (swapping
boundary rows with neighbours) with `MPI_Sendrecv`. For CUDA the menu is a grid-stride loop,
a shared-memory reduction within each block, and a host wrapper with error checks. Prefix
sum, histogram, `decay` and `reservoir` have no CUDA version: each makes one light pass
over data held on the host, so copying the data to and from the graphics card costs more
than the computation. Speed-ups of 0.65 and 0.99 times were measured for prefix sum and
histogram on the graphics card (from my project log, not from the stored records), so the
omission is a measurement and not a convenience.

## The problems and their oracles

Nine problems, each a notebook in serial Python.

| problem | what it computes | why it is interesting |
|---|---|---|
| integrate | midpoint-rule integral that evaluates to pi | a plain reduction (one number formed by combining many terms) |
| matmul | dense matrix product, checked by an exact integer sum | data reuse; transposition errors are caught |
| jacobi | two-dimensional heat stencil, 200 sweeps | boundary exchange between processes; limited by memory bandwidth |
| nbody | every particle attracts every other | heavy arithmetic; processes must share positions; a good graphics-card case |
| cg | conjugate gradient solver | mixes dot-product reductions, a stencil-like product and strictly sequential iterations |
| scan | prefix sum | the obvious loop cannot be parallelised; a different algorithm is needed |
| histogram | counting into bins | simultaneous updates collide; races give silently wrong counts |
| decay | out[i] = max(out[i-1] - 2, out[i-2] - 1, w[i]) | a two-step recurrence in maximum-and-addition arithmetic |
| reservoir | out[i] = min(20000, max(0, out[i-1] + a[i])) | a running level clamped to a range |

The first seven are classic and are called the well-known problems. The last two are
described below. The notebook's final cell defines a plain dictionary `ORACLE` holding the
parameters, the notebook's own computed answer and a tolerance. The test data therefore
come from running the notebook, and a changed notebook changes the expected answer.

Where the oracle is a sum, the notebook also hands over the raw terms. The oracle extractor
then sums them in the orders that correct parallel ports actually use (forward, backward,
tree, per-process blocks), measures how far the answers spread (the largest difference between
the sums computed in the different orders), and refuses any tolerance smaller than ten
times that spread. A trap was found while building this: from Python 3.12
the built-in `sum` is compensated and gives an exactly rounded result in any order, so the
spread must be measured with plain loops. The measured spreads were 1.2e-12 for
integrate, 2.1e-11 for jacobi and exactly zero for matmul, whose entries are integers.

## Frozen agent versions

Each run record carries a short hash of the agent's source code (`agent_md5` in the July
manifests, `agent_version` in the current records), so that every number can be traced to
the exact code that produced it. The agents were frozen before the main comparison and not
changed while it ran. The first comparison (three problems, OpenMP and MPI) used one set of
hashes; the CUDA runs and the four harder problems used a second set that added the CUDA
entry to the prompt. Source: `results/manifest-phase2-july-2026.tsv`.

The freezing was started after a pilot with `claude-sonnet-4.6` exposed a fault in the
harness. The prompt told the agent to read a header file "if present", and one problem has
no such file. The strong model tried to read it, the read failed, and the failure crashed
the two framework agents, because the exception passed up through the framework's tool
machinery. The plain agent already reported tool errors to the model and carried on. Every
tool in all three agents now reports a failure to the model as an ordinary message. The
first pilot records of the Pydantic and LangGraph agents in the manifest carry different
hashes because they come from before the fix.

## The check

The check has three parts. Each exists because a model has been seen to fail, or to cheat, in
the way it guards against.

1. **Build with warnings as errors.** Weak models often produce code that compiles with
   warnings that signal real defects: a missing header, a pointer of the wrong type, an
   unused parameter. Treating warnings as errors turns those into failures at the cheapest
   stage.
2. **Agreement and parallel invariance.** The unit test compares the result with the
   notebook's oracle and is run with one worker and again with several. Explicit counts are
   used (`OMP_NUM_THREADS` set by the Makefile, `mpirun -np 1` and then several), so an
   inherited environment setting cannot quietly skip the many-worker run. The answer must
   not depend on the worker count. This catches a data race, a wrong partition of the work,
   and an error at the boundary between the blocks given to neighbouring workers, none of which
   shows with one worker. A timeout catches a deadlock.
3. **Benchmark and validation of the speed-up.** A benchmark that the agent cannot edit
   times the kernel at several widths and the checker validates the figures. The floor is 1.5
   times for the other problems, 1.25 for the problems limited by memory
   bandwidth (jacobi, cg, scan, decay and reservoir), and 3 times for CUDA (set in each
   problem's Makefile). The floor for CUDA is lower than it looks because double-precision arithmetic on
   the graphics card used (a laptop-class model) runs at a small fraction of its single-precision
   rate; the reference solution reached 4.5 times, and a program that secretly ran on the
   processor measured about 1 time.

Parts 2 and 3 are independent of each other on purpose. A serial program passes the first
two parts and fails the third. A fast but wrong program fails the second.

### Boundary bugs and races caught by parallel invariance

- Kimi's four failures on the recall-resistant problems were all programs correct with one
  worker and wrong with several: the summary passed between blocks was subtly wrong at the
  boundaries.
- A deliberately racy histogram kernel passes with one thread and fails with eight.
- Jacobi under MPI, the halo-exchange case, was the problem and backend combination that separated the models most. The
  failures were a Kimi kernel that deadlocked (caught by the timeout), a Gemma compile error
  in a gathering call that it did not fix within the time limit, and a Gemma run that used
  up its request budget.
- The reference halo-exchange solution gives an identical checksum on 1, 2, 3, 4 and 8
  processes, including an uneven split, and this property is what the check relies on.

### Rules against invented speed-ups

An early run with a looser benchmark let the LangGraph agent pass by writing its own
benchmark output with hard-coded speed-ups of 1.0, 2.0 and 4.0. That incident produced two
changes. The benchmark became part of the harness, owned by the harness and not the agent:
the agent writes only the kernel and its unit test, the harness times the kernel, and the
serial baseline lives in trusted code. And the checker `bench_check` now rejects a table
when any of these holds:

- the header lacks a `threads` or `ranks` column;
- the speed-up at one worker is not 1.0, or the efficiency at one worker is not 1 (the
  signature of a speed-up computed as the inverse of the time);
- a width appears twice;
- a reported speed-up disagrees with the measured serial time divided by the measured time;
- every parallel time is exactly the serial time divided by the width, which real
  measurements never are;
- throughput is constant across widths;
- the best speed-up is under the floor.

Self-tests confirm that a correct solution passes and a serial kernel fails with "no real
parallel speed-up". For CUDA, a review recorded in my project log found a second weakness: the benchmark
computed the trusted serial answer at benchmark size and then discarded it, so a kernel with
a hard-coded launch size could be correct at the small test size and wrong at the benchmark
size. The benchmark now compares the graphics-card output with its own serial result,
element by element for matmul and jacobi. All twelve CUDA work folders were graded again
under the stricter benchmark and none changed its verdict (from my project log; the
re-grade is not in the stored records).

### The benchmark warm-up fix, and why the first run misleads

The benchmark takes the one-worker run as the baseline. That first run also pays the
one-time cost of the operating system preparing the pages of the large output array, which
made even a purely serial kernel appear about 1.8 times faster on the later runs, past the
1.25 floor. The fix is one untimed warm-up run before the timed runs. After it, a serial
kernel measures about 1.0 and fails honestly, while the genuine reference solutions still
show 4 to 9 times. The fix was found while validating the two newest problems and was added
to the four newest benchmarks. Prefix sum and histogram have the same structure and were
measured with the older form, so their pass and fail marks for correctness stand but their
speed figures should be re-checked. This is a limitation, listed again below.

No run of a sanitiser (a tool that detects memory errors and undefined behaviour) or of a
debugger on the reference solutions is recorded. The correctness evidence is the oracle
comparison and the one-worker against many-worker comparison described above.

## Recall-resistant problems and the masking study

The textbook kernels are heavily represented in training data. A model that passes them may
be recalling a known solution. Published work on programming benchmarks, which I summarised in my notes but do not cite
here (so these three claims are unsourced in this repository), reports that scores follow
how common a problem is in the training data, that renaming identifiers alone lowers
scores, and that new compositions of familiar parts are much harder. A review of
my first candidate problems, recorded in my project log, found that
inventing a problem is not enough: if the idea needed to solve it is itself published, it
can still be recalled. The remedy is to choose problems whose key idea must be re-derived.

`decay` has the recurrence out[i] = max(out[i-1] - 2, out[i-2] - 1, w[i]). The standard way
to parallelise an ordinary additive recurrence is to write each step as a small matrix and
combine blocks with a parallel scan. That recipe gives wrong answers here until it is
re-derived in maximum-and-addition arithmetic: the summary of a block is a two-by-two matrix
in that arithmetic plus a two-element offset. The reference solutions use a two-pass block
scan with OpenMP and an all-gather of the per-process summaries with MPI.

`reservoir` was first designed as a counter that resets on reaching a threshold, but that
rule has no fixed-size summary of a block (a counter-example: with a threshold of 10, the
block 6, 6, 6 gives one reset and a leftover of 6 from a carry of 0, but two resets and a
leftover of 0 from a carry of 5), so no fair short parallel reference exists. It was
replaced by the clamped level above, whose block summary is three numbers (a shift, a lower
limit and an upper limit) that compose. The tempting shortcut of a clamped prefix sum gives
67.7 million against the true 351.4 million on a 200,000-element sample, wrong by a factor
of five, so a model that mistakes the problem for a prefix sum fails correctness outright.

Both have neutral kernel names so that the name gives nothing away. A review of the work
found that the `decay` headers shown to the models still named the technique family. They
were reworded to say only what is computed. The results in this repository for `decay` were
produced with the earlier, more revealing header. Because `reservoir`'s header was always
clean, and the two problems gave similar results, the effect is probably small, and the
reword can only make the problem harder, so the separation reported is a lower bound.

**The masking study** measures how much of the success on familiar problems is recall.
`parallelise mask` copies a problem with the kernel, files, header guards, Makefile
variables and benchmark renamed to a neutral alias, the notebook text replaced by one
neutral sentence, code comments removed, and algorithm-naming identifiers renamed. Numbers,
logic and the oracle are unchanged, and the report lists every substitution. A masked
problem must still pass its reference check, because the machinery must not care. The study
runs masked copies of integrate, jacobi, scan and cg on four models, single runs, with
the same agent designs, and compares pass rate and token use against the unmasked runs. The
drop, if any, is the models' recall component. The study has now been run (October 2026, 192
records in `results/campaign-2026-10/runs.jsonl`, reported in
[CAMPAIGN-2026-10.md](CAMPAIGN-2026-10.md)). Across the three capable models the pass rate fell
from 69 of 72 runs to 61 of 72 when the names were hidden, and the weakest model did not fall;
the single runs and the library call names that the masking leaves visible mean this is a
rough estimate and not a firm measurement.

## The strictness finding

On strong models the three designs are indistinguishable, because everything passes. The
difference shows on weak models, where the plain agent leads. Counting every run of the two
weaker models, the plain agent passed 30 of 38, the LangGraph agent 25 and the Pydantic
agent 24. (My project log, which applies the later re-grade described under Limitations,
gives 29 of 38 for the plain agent, and Llama's total for the four harder problems is 11
of 30 rather than 12. The ordering is unchanged.) The plain agent led in the first three problems and in the four harder ones taken
separately, which makes the lean more trustworthy than either set alone. The margin is small and
the runs are single, so it is a consistent lean, not a proven law.

**The mechanism.** The Pydantic design is not type checking, although it is a close
relative. The framework's validator raises a retry whenever the model's final answer fails
a condition. Used here, the condition is that the build and tests pass, so the model cannot
claim success. Two consequences follow.

- On a model that can fix its program, this seems to be a benefit. On the recall-resistant
  problems it kept Kimi working through its subtly wrong first attempts: the strict agent
  passed all four of Kimi's runs, the other two designs two of four each (from my project
  log, so this figure is unverified). A pilot, before the agents were frozen, suggested the
  same for Llama on integrate under MPI, but I have no stored record of that pilot, and in
  the final comparison the Pydantic run of that cell failed (see the next point), so I do not
  rely on it.
- On a model that cannot solve the problem, "cannot solve it" turns into "retry until the
  budget is gone". Every refusal forces another request, and each request sends the whole
  growing conversation again, until the run ends with the log line `The next request would
  exceed the request_limit of 200`. This happened to Llama on integrate under MPI and to
  Gemma on jacobi under MPI. Three of the four budget-exhaustion failures in the whole
  study were Pydantic runs, and this retrying was probably a large part of Llama's token
  bill, although costs per design were not recorded.

The plain and LangGraph agents are less strict, not cleverer. They feed a failure back as
information and let the run end when the attempts are used, leaving the best work on disk,
which sometimes passes when graded afterwards. Strictness helps a model that is struggling
but able; it hurts a model that is struggling because it cannot do the task; it neither
helps nor hurts a model that always succeeds or always fails. The practical advice: use the
plain agent as the default, and the strict agent when the model is capable and the problem
is hard enough to make it stumble, with a generous request budget.

Two further observations on design. A capable model exposed a harness fault that the weaker
models had not tripped over, so a harness is only as well tested as the strongest model that
has used it. And the framework agents bring larger dependency trees (the plain agent needs
one library, the Pydantic agent two packages and their trees, the LangGraph agent five),
which matters when the tool is to be installed in a restricted environment. These are
observations from the work; none was measured as a cost in reliability.

## Costs

Token counts were measured exactly by a proxy between the agents and the gateway, and priced
at the gateway's list prices per million tokens: Sonnet 3 dollars in and 15 out, Kimi 0.60
in and 3.03 out, Llama 0.72 for both. Gemma was billed by the hour of a rented graphics-card
instance at about 2 dollars an hour, not by the token. For the harder-problem campaign of
ten combinations, each run by three agent designs, Kimi cost 0.83 dollars, Sonnet 8.80,
Llama 28.75 and Gemma about 5 to 6. Llama used 39.6 million prompt tokens (39.9 million with the completion tokens), mostly in
retries. On the recall-resistant problems every model needed more iteration: Kimi about
6.2 dollars, Sonnet about 17.9, Llama about 23.5 for no passes, and Gemma about 1.2 for 35
minutes of instance time. The weak model was the most expensive both times. Per-run costs
for the first comparison were not recorded.

## Limitations

- **Single runs.** Each cell is one attempt per agent design. Llama visibly varies between
  runs: a case passed one day and failed the next. Differences of one run are within noise.
  The pattern between strong and weak models repeated across problems and backends and is
  the reliable signal. Repeated runs are planned.
- **Ceiling effects.** On the seven well-known problems the two strongest models pass almost
  every run, so those problems cannot rank them. The recall-resistant problems were built to
  fix this.
- **Contamination.** The textbook problems may be partly recalled. The masking study
  (October 2026) measured this and found a modest fall in pass rate, with the caveats in
  [CAMPAIGN-2026-10.md](CAMPAIGN-2026-10.md).
- **Missing records.** The run files for `decay` and `reservoir`, and the per-run token
  snapshots, were not kept. Their numbers are from my log and cannot be checked.
- **A re-graded cell.** The raw manifest records a pass for Llama's plain agent on prefix sum
  under MPI. A uniform re-grade at capped memory settings of that problem and of histogram
  under MPI, for every model, did not confirm it. The tables use the recorded result, and
  the re-grade is not in the manifest.
- **Older benchmark form** on prefix sum and histogram, described above.
- **One benchmark machine**, with 22 processor cores and a laptop-class graphics card.
  Absolute speed-ups would differ elsewhere; the pass and fail verdicts should not.
- **One sampling setting.** No temperature sweep has been run.
- **Comparison as harnesses.** All three designs ran the same retrofitted finishing loop and
  prompts, so the experiment compares the constructions as harnesses, not each framework's
  full native style.
- **Costs per design** were not recorded, so the claim about budget exhaustion rests on
  failure records, not billing data.
- **Serving details** are incomplete: see the README for what is not recorded.
