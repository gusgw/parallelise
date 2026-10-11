/* scan.c — parallel inclusive prefix sum (two-pass block scan, OpenMP).
 *
 * Algorithm (as specified in scan.h):
 *   Pass 1 : each thread sums its contiguous block of 'in' into block_sum[tid].
 *   Serial : exclusive scan of block_sum[] gives each block's starting offset.
 *   Pass 2 : each thread scans its own block of 'in' into 'out', starting from
 *            its offset.
 *
 * This splits the work across threads so each thread touches only its share of
 * the array in both passes; no thread repeats the whole computation.
 */

#include <stdlib.h>   /* malloc, free */
#include <omp.h>      /* omp_get_num_threads, omp_get_thread_num */

#include "scan.h"

void scan(const double *in, double *out, long n)
{
    if (n <= 0) return;

    /* ------------------------------------------------------------------ */
    /* Determine the number of threads that will actually be used.         */
    /* We query inside a parallel region so the runtime's own decision     */
    /* (which may be capped by OMP_NUM_THREADS) is respected.              */
    /* ------------------------------------------------------------------ */
    int nthreads;
#pragma omp parallel
    {
#pragma omp single
        nthreads = omp_get_num_threads();
    }

    /* Per-block partial sums and exclusive-scan offsets. */
    double *block_sum    = (double *)malloc(sizeof(double) * (size_t)nthreads);
    double *block_offset = (double *)malloc(sizeof(double) * (size_t)nthreads);
    if (!block_sum || !block_offset) {
        /* Fallback: serial scan so the result is still correct. */
        free(block_sum);
        free(block_offset);
        double run = 0.0;
        for (long i = 0; i < n; i++) { run += in[i]; out[i] = run; }
        return;
    }

    /* ------------------------------------------------------------------ */
    /* Pass 1 : each thread sums its contiguous block.                     */
    /* ------------------------------------------------------------------ */
#pragma omp parallel num_threads(nthreads)
    {
        int tid  = omp_get_thread_num();
        int nt   = omp_get_num_threads();

        /* Divide [0, n) into nt contiguous blocks as evenly as possible. */
        long block = (n + nt - 1) / nt;   /* ceiling division             */
        long start = tid * block;
        long end   = start + block;
        if (end > n) end = n;

        double s = 0.0;
        for (long i = start; i < end; i++) s += in[i];
        block_sum[tid] = s;

        /* ----------------------------------------------------------------
         * Serial exclusive scan of block_sum[] → block_offset[].
         * Only one thread does this, after all threads have written their
         * partial sums.
         * ---------------------------------------------------------------- */
#pragma omp barrier
#pragma omp single
        {
            double acc = 0.0;
            for (int t = 0; t < nt; t++) {
                block_offset[t] = acc;
                acc += block_sum[t];
            }
        }
        /* implicit barrier after 'single' ensures block_offset is visible */

        /* ----------------------------------------------------------------
         * Pass 2 : each thread scans its own block, starting from its
         * offset.
         * ---------------------------------------------------------------- */
        double run = block_offset[tid];
        for (long i = start; i < end; i++) {
            run  += in[i];
            out[i] = run;
        }
    } /* end parallel */

    free(block_sum);
    free(block_offset);
}
