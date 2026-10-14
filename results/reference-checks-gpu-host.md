# Reference checks on a 12-core host with a 16 GB NVIDIA graphics card

Date of the runs: 11 October 2026.

## Purpose

Until this run, no machine I had access to could execute every part of the reference
checks: the message-passing (MPI) reference solutions had not all been timed on a machine
with more than a few cores, the CUDA reference solutions had not been built in the
container, and the masked MPI reference had never been run through the check. This page
records what happened when each reference solution was passed through the check on a single
host that has all the tools. The check is described in
[docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

## The host and the images

- The host has 12 processor cores, 30 GB of memory and an NVIDIA RTX 5060 Ti graphics card
  with 16 GB of memory. It has Docker with the NVIDIA runtime, but no MPI compiler and no
  CUDA compiler, so every MPI and CUDA build ran inside the repository's container images.
- `parallelise:cpu`, built from the `cpu` target of the `Dockerfile`, was used for the
  OpenMP and MPI runs. `parallelise:cuda`, built from the `cuda` target (based on the NVIDIA
  CUDA 12.8.0 development image on Ubuntu 24.04), was used for the CUDA runs, started with
  `--gpus all`.
- Both images build without error. The tests ran as the image's own unprivileged user, with
  nothing else heavy running on the host. The speed-up floor was not relaxed:
  `PARALLELISE_SPEEDUP_UNRELIABLE` was not set. Open MPI oversubscription is enabled in
  the images but was not needed, because no run used more than 12 ranks.
- Each figure below is one run of the check. The command was
  `python -m pytest -q -rs -m "slow or mpi" -k reference` for OpenMP and MPI, and
  `python -m pytest -q -rs -m cuda -k reference` for CUDA. Because the test report does
  not show the speed-ups, I then ran the same check on each case a second time with a small
  script that prints the benchmark table, and the speed-ups below come from that second run.
  Every case passed in both runs. Each speed-up is the best over the worker counts 1, 2,
  4, 8 and 12 (or the GPU against one CPU core), measured against the one-worker run of
  the same benchmark.

## Outcome

All 23 reference solutions passed: 9 OpenMP, 9 MPI and 5 CUDA, plus the masked checks
further down. The pytest summaries were "20 passed, 5 skipped" for the OpenMP and MPI
command (the 5 skips were the CUDA cases, which cannot build in the CPU image) and
"5 passed" for the CUDA command.

| problem | backend | result | best speed-up | floor |
|---|---|---|---|---|
| conjugate gradient | OpenMP | pass | 4.59 | 1.25 |
| conjugate gradient | MPI | pass | 3.51 | 1.25 |
| conjugate gradient | CUDA | pass | 17.60 | 3.0 |
| decay | OpenMP | pass | 6.58 | 1.25 |
| decay | MPI | pass | 7.05 | 1.25 |
| histogram | OpenMP | pass | 3.09 | 1.5 |
| histogram | MPI | pass | 3.01 | 1.5 |
| integration | OpenMP | pass | 5.30 | 1.5 |
| integration | MPI | pass | 3.92 | 1.5 |
| integration | CUDA | pass | 8.95 | 3.0 |
| heat stencil (Jacobi) | OpenMP | pass | 6.18 | 1.25 |
| heat stencil (Jacobi) | MPI | pass | 5.42 | 1.25 |
| heat stencil (Jacobi) | CUDA | pass | 11.51 | 3.0 |
| matrix multiply | OpenMP | pass | 5.05 | 1.5 |
| matrix multiply | MPI | pass | 8.95 | 1.5 |
| matrix multiply | CUDA | pass | 27.53 | 3.0 |
| N-body | OpenMP | pass | 6.03 | 1.5 |
| N-body | MPI | pass | 5.72 | 1.5 |
| N-body | CUDA | pass | 15.23 | 3.0 |
| reservoir | OpenMP | pass | 7.02 | 1.25 |
| reservoir | MPI | pass | 5.80 | 1.25 |
| prefix sum (scan) | OpenMP | pass | 1.93 | 1.25 |
| prefix sum (scan) | MPI | pass | 1.32 | 1.25 |

CUDA was built for five problems only; there is no CUDA version of prefix sum, histogram,
decay or reservoir. The CUDA speed-up is the serial C baseline in the benchmark divided by
the time on the graphics card, so it is a speed-up over one CPU core.

## Notes

- The conjugate gradient OpenMP reference had missed its floor on an 8-thread laptop. On
  these 12 cores it passed comfortably, with a best speed-up of 4.59 against a floor of
  1.25. The test file still lets that one problem turn a missed floor into a skip; no skip
  was needed here.
- Speed-up generally peaked at four to eight workers and then fell at twelve, as the
  benchmark sizes are small and the work is limited by memory bandwidth or by message
  passing. Prefix sum with MPI passed with the narrowest margin (1.32 against 1.25), and
  that margin is small enough that a busier host could fail it.

## The masked reference

I masked the prefix sum problem with `parallelise mask scan kernel_a <destination>`, which
made 16 renaming substitutions, and passed the masked reference solutions through the check
in the CPU image. This is the first time a masked MPI reference has been run.

| problem | backend | result | best speed-up | floor |
|---|---|---|---|---|
| masked prefix sum (`kernel_a`) | MPI | pass | 1.54 | 1.25 |
| masked prefix sum (`kernel_a`) | OpenMP | pass | 1.94 | 1.25 |

## Failures and the one fault found

No reference solution failed on this hardware. One fault was found and fixed in the
repository: the first CUDA check run failed at the build stage for all five problems,
because each CUDA `Makefile` defaults to a compiler path that exists only on the author's
workstation (`/opt/cuda/bin/nvcc`), and that path does not exist in the container. The
`Makefile` setting can be overridden from the environment, so the fix was to add
`NVCC=/usr/local/cuda/bin/nvcc` to the environment of the `cuda` stage of the `Dockerfile`.
The image was rebuilt and every CUDA reference then passed. The problems themselves were
not changed.
