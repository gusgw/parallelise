# Backend assessments of the nine notebooks, November 2026

These 36 files are the output of `parallelise assess`, run on 6 November 2026 on each of the
nine notebooks under `problems/` with two models, `kimi-k2.5` and `claude-sonnet-4.5`, both
served by Amazon Bedrock through a LiteLLM gateway at the default temperature of 0.2. Each
notebook and model was assessed once. For each pair, `<problem>-<model>.json` is the verdict
as the command wrote it with `--out`, holding the recommended backend, the parts of a
combination, the reasoning fields, the model's full reply in `analysis`, and the tokens and
time of the call; `<problem>-<model>.txt` is what the command printed: the full reply and
the closing line that names the recommendation.

The recommended backend of each verdict:

| notebook | claude-sonnet-4.5 | kimi-k2.5 |
|---|---|---|
| cg (conjugate gradient) | OpenMP | OpenMP |
| decay | serial | serial |
| histogram | OpenMP | OpenMP |
| integrate (midpoint integration) | OpenMP | OpenMP |
| jacobi (heat stencil) | OpenMP | OpenMP |
| matmul (dense matrix multiply) | OpenMP | combination: OpenMP for small and medium sizes, CUDA for large sizes with a graphics card, MPI for large sizes on a cluster |
| nbody (N-body forces) | OpenMP | combination: OpenMP for the forces, with a CUDA variant |
| reservoir | serial | serial |
| scan (prefix sum) | OpenMP | CUDA |

For a combination, the single backend that the harness would run is the first part that is
not serial, which is OpenMP for both of Kimi's combinations.

The eighteen calls used 10,259 prompt and 9,373 completion tokens for `claude-sonnet-4.5`
and 9,234 and 8,234 for `kimi-k2.5`, about 0.17 and 0.03 US dollars at the list prices in
`results/campaign-2026-10/prices.json`. The comparison of these verdicts with the measured
speed-ups of the known-good solutions is in `docs/RESULTS.md`.
