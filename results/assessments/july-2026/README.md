# Backend assessments of three notebooks, July 2026

These twelve files are the output of the first version of the backend-assessment stage, a
stand-alone script that was later ported into the package as `parallelise assess`. The script
gave a whole notebook to a model and asked for a critical analysis and a recommendation among
OpenMP, MPI, CUDA, a combination, or serial code. Each notebook was assessed by two models.
For each pair, the `.txt` file is the model's full reply and the `.json` file is the verdict
extracted from it. The files use the July verdict format, whose first field is called
`primary`; the package now writes `backend` and reads either.

| notebook | claude-sonnet-4.6 | kimi-k2.5 |
|---|---|---|
| integrate (midpoint integration) | OpenMP | OpenMP |
| jacobi (heat stencil) | OpenMP | OpenMP |
| matmul (dense matrix multiply) | OpenMP | CUDA |

How the verdicts compare with what was later measured, at the problem sizes used in the
study: both models were right to call CUDA a poor fit for integrate, where the loop does
very little work per element; neither recommended MPI for any of the three, and MPI was
indeed the weakest fit at these sizes; for matrix multiply, the one problem where the GPU
gave by far the largest measured speed-up (about 28 times over one processor core on the
GPU host, against 5 to 9 times for OpenMP), Kimi's verdict matched the measurement and
Sonnet's did not. Two models on three notebooks is a trial, not a validation; validating
the stage against the measured results at scale is the next experiment.
