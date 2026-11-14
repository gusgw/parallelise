# Design

parallelise asks a large language model, working through an agent that gives it a task and some
tools, to write a parallel version in C, called the routine, of a NumPy function held in a Jupyter
notebook. The routine is for one of three backends, or ways of running in parallel: OpenMP (Open
Multi-Processing, threads within one computer), MPI (the Message Passing Interface, processes that
exchange messages) or CUDA (NVIDIA's platform for computing on graphics cards). The harness, the
program that sets each task, judges whether the routine is correct and whether it is faster than
plain serial C, so that models and designs of agent can be compared. The [README](../README.md)
says how to use it and [RESULTS.md](RESULTS.md) reports what the runs found; this document gives
the reasons for the design.

Two principles shape the design. Each agent has a narrow task and sees only what that task needs,
so that no agent has both the means and a reason to cheat. And every verdict comes from fixed rules
in code, never from a model's opinion. No code is aimed at a particular cheat: the cheats found so
far fail because of how the steps are arranged.

## The serial routine

First, an independent agent, which knows nothing of the rest of the work, translates the function
into plain serial C. It is told to translate faithfully, with no word of parallel work, and it uses
one model for every problem, at temperature zero, the setting that makes a model's output least
random. The translation becomes the problem's `serial.c` only if it agrees with the Python on fresh
inputs at every test size and at the benchmark size.

The serial routine referees correctness at the benchmark size, where the Python is often too slow,
and it sets the baseline of every speed-up. A baseline taken from the routine's own run with one
worker could be slowed on purpose; one written by an agent that knows nothing of the goal has no
reason to be slow or fast, and does not change with the model under study.

## What the agent sees

The model under study may read three files besides its own routine, all masked in a study run: the
function and what it uses, without the notebook's demonstrations, which can give away what it
computes; the serial C, so that skill at parallel work is not confused with skill at translation;
and a header generated from the declaration. It writes no tests, runs no commands, can change no
file but its routine, and never sees an expected value or the code that judges it. It may finish
only when its check finds the routine correct and at least 1.2 times as fast as the serial routine,
a goal that plays no part in the verdict.

The header states one contract for every problem. Every process receives all the arguments, and
when the routine returns its results must be complete, on process 0 for MPI and in ordinary memory
for CUDA. A generated driver program reads the inputs, and for MPI shares them among the processes,
before the call, so that neither is timed; gathering the results and copying to and from the
graphics card are the routine's work, and are timed.

## The rule

A rule on what a routine may contain is applied before every build, in the agent's check as in the
judgement, and it binds the serial routine too. The routine may read only its own header and the
system's headers, and this is checked before it is compiled, so that no message of the compiler can
show the agent another file. Its code, with its macros expanded, may hold no assembly language,
through which it could read a file that the compiler would not list. It may define no name but its
own, so that it cannot replace a function that the driver or a library needs. It may call only the
C library's mathematics, memory and printing functions, a few others such as `qsort`, and its
backend's calls other than clocks, file functions and those that start or connect to other
processes. It may make no system call and hold no code that runs before the program starts. Under
CUDA the tests of defined names and of start-up code are relaxed, because the CUDA compiler adds
names and start-up code of its own. The routine must also compile with warnings treated as errors.

## The judgement

After the agent stops, the harness judges the routine in a fresh folder that the agent never saw.
It tests correctness on fresh random inputs at the test sizes and at two sizes drawn between them,
which the agent could not foresee, each in a fresh process, so that nothing carries over between
calls. Each input runs once with one worker and once with all of them, which can expose a race or a
wrong division of the work. A call at a test size that takes more than two minutes counts as hung,
and fails. All of them means one for each processor core that the harness may use, counting a core
once however many hardware threads it has, since a routine with a worker on every hardware thread
waits at each step whenever any other program runs; MPI starts fewer processes if their copies of
the data would not fit in 6 GiB (gibibytes); and a CUDA routine runs once. Only then does the
Python compute the trusted answers, so that they do not exist while the routine runs. Every result
must have the declared type and shape, be finite and agree within its tolerance.

At the benchmark size the routine with all its workers, the serial routine and, except under CUDA,
the routine with one worker each run in a process of their own and take turns on six new inputs, so
that a change in the machine's speed falls on all three alike, and every answer is checked against
the serial routine's. The first round is not timed, to keep one-time costs out of the measurement.
Because the two routines run on the same inputs, they are compared input by input, and inputs that
differ in their amount of work, as `jacobi` and `cg` draw their number of steps afresh, cannot hide
a gain. The speed-up is the median, over the five timed inputs, of the serial routine's time
divided by the routine's, and the routine counts as faster only if it beats the serial routine on
all five, which chance alone would give once in 32 trials if the two were equally fast. Its
scaling, the median ratio of its time with one worker to its time with all, is reported too, and it
is called parallel if all its workers beat one on all five inputs, or, under CUDA, if it starts work
on the graphics card.

The harness times each call from outside the routine's process. The driver writes a mark to a named
pipe, a channel between two programs, just before and just after the call, and the harness reads
its own clock as each mark arrives; since the routine may open no file, it cannot forge a mark.
After the first mark the driver waits for the harness's answer before it calls the routine, so that
a routine that fills every processor cannot delay the reading of the clock and so shorten its own
time. Between calls the processes of an MPI routine wait asleep, so that they take no processor
time from the serial routine while it is timed.

## Cheats

Under the earlier design, agents ran commands in the same folder as the files that judged them, and
in October 2026 four runs changed some of those files; they have the failure stage `tampered` in
`results/campaign-2026-10/runs.jsonl`. Deliberate attempts to fool that design's check later
produced three routines, now kept in `tests/fixtures/cheats/`, and none of them deceives the
present design. One summed only a tenth of its subintervals when there were more than a million,
and it fails because every call at the benchmark size is compared with the serial routine. One was
serial and repeated its work when it had few threads, to make its one-thread run look slow, and
gains nothing, because the baseline is not its own. One kept its answer between calls, and meets a
new input on every call, besides being refused for reading the clock.

The October routine for the prefix sum that the README describes read its checksum from a header of
expected answers
(`results/campaign-2026-10/runs/masked/pydantic/scan/mpi/gpt-oss-120b/pydantic/kernel_c.c`). It
passed then and in a re-check in November. Judged again now, it does not even build, because no
expected value is within its reach; and if it did, its zeros would fail a comparison of every
element, with trusted answers that do not exist while it runs.

Some cheats remain open. A routine may do less work than asked where the tolerance allows: the
midpoint rule converges so fast that one using a four-hundredth of its subintervals at the
benchmark size would stay inside the integration problem's tolerance, although that tolerance is
only about twenty times the differences that honest orders of addition make. A routine that slows its own run with one worker inflates its
scaling, though not its speed-up. A routine wrong only on inputs never drawn, or with a race that
rarely shows, can pass by luck. The reverse conversion, below, is meant to expose most of these.

## The reverse conversion

A last step has an independent agent translate the parallel routine back into plain Python,
statement by statement. The agent sees only the routine, renamed `routine` and without its comments;
it never sees the notebook, the problem's name, the goal or any verdict. It must take the number of
workers as an argument and translate every statement, including those that change no result. Its
check compares its Python with the C on fresh inputs, never with the notebook, so that nothing
steers it towards hiding a trick, and it names each result by its place among the routine's
parameters, so that it gives away no declared name.

A fixed rule then reads the translation and flags anything that a faithful numerical routine has no
need for: an import other than math and NumPy, a file, a clock, the environment, state kept between
calls, a call to code that the translation does not define, a loop that only waits, a write to an
argument that the routine may only read, or a comparison of the number of workers with a fixed
number. The routine is also checked against the notebook at every size where the translation shows
its code changing, from the fixed numbers that its sizes are compared with. A repeat with one worker
or a kept answer thus shows as an obviously silly serial algorithm, and a shortened loop as a wrong
answer at the size where it begins. When anything is flagged, the fixed model is shown the
notebook's Python and the translation, and nothing else, and asked whether the two compute the same
results; its answer goes to the reader beside the flags, never into a verdict. In a trial on 9
November, with Kimi K2.5 as the translator, the rule flagged the three cheats above, each for the
trick it hides, and none of the 39 known-good routines, whose translations all agreed with their C.
A first trial, of an earlier version, had wrongly flagged one known-good routine, through a fault in
the rule since mended and a loop that the translator wrote and the C does not have. Of the 297
stored routines from July and October that the judgement finds correct, it flagged 12, as RESULTS.md
reports, 11 of them for comparing the number of workers with 0 or 1, or testing whether a size
divides by it. A race has no serial form, and remains a matter for testing.

## Agent designs, the meter and masking

Three agent designs are compared: a plain loop written directly against the model's web interface,
and two built on the frameworks Pydantic AI and LangGraph. They share one prompt, one set of tools
and one rule for stopping, enforced in each design's own way: a run ends when the check passes,
when the budget of requests is spent, when the time limit passes or when the model service fails.
Every request to the model passes through one meter, which times each answer, counts its tokens
(the pieces of text that a model reads and writes) and waits out the refusals of a busy service, so
that all three designs are measured alike.

Masking renames every name that the sources define, the same in the Python, the C and the header,
and removes the comments and docstrings, except the header's statement of the contract, so that a
model cannot recognise an algorithm by its name and recall a solution. The agent's check builds the
routine against the same masked header and calls the arguments and results by their masked names,
so that neither a compiler's message nor the report of a wrong answer gives the old names away.
Names that belong to Python, NumPy, the C library or the backend stay, so the masked prefix sum
still calls `np.cumsum`: masking makes recall harder but cannot prevent it.

## The problems

The thirteen problems under `problems/` are a sum (`integrate`), a matrix product (`matmul`), a
stencil (`jacobi`), forces between all pairs of bodies (`nbody`), conjugate gradients for a given
number of steps (`cg`), a running total (`scan`), counts in bins that several workers may update at
once (`histogram`), two recurrences whose parallel form, a running total under an unusual
operation, must be worked out afresh (`decay` and `reservoir`), Monte Carlo sampling (`ising`),
molecular dynamics compared through averages (`md`), a sparse solver that stops when it converges
(`sparse_cg`), and a result whose length depends on the data (`compact`). The tolerance of a sum is
typically about ten times the largest difference measured between the Python and an honest routine
that adds in another order; averages are compared within five times their statistical spread, and
whole numbers exactly. A known-good routine for each problem and backend, 39 in all, kept under
`tests/fixtures/known_good/`, tests the harness; all pass, and none judges a run.

## Limits

The serial routines are plain translations, and some are slower than careful serial code: the
matrix product reads its second matrix by columns, which costs a factor of about nineteen, and the
routines for conjugate gradients, the Ising model and molecular dynamics take up to twice as long as
they need. Speed-ups against them are larger than against careful serial code, which is one reason
the scaling is reported. At the benchmark size each program keeps one process for all six inputs,
so that one-time costs stay out of the timed calls; new inputs defeat a stored answer, but a
routine could keep other state between those calls against its contract. Timings mean something
only on a quiet machine, so judgements should be made one at a time, and each judgement's log
records the machine's load average around the benchmark, so that one made beside other programs can
be found afterwards. Routines run with the user's rights and in no sandbox, though without the
environment variables that may hold secrets; the rule limits what a routine contains, but does not
prove that it cannot reach outside itself.
