/* scan.c — parallel inclusive prefix sum (two-pass block scan, OpenMP).
 *
 * Algorithm (as described in scan.h):
 *   Pass 1: each thread independently sums its contiguous block of `in[]`
 *           and stores the block total in block_sum[tid].
 *   Serial: exclusive scan of block_sum[] gives each block's starting offset.
 *   Pass 2: each thread scans its own block of `in[]`, starting from its
 *           offset, writing the inclusive running total into `out[]`.
 *
 * This is the classic shared-memory parallel prefix sum; every element of
 * `in` is read exactly twice (once per pass) and every element of `out` is
 * written exactly once.
 */

#include "scan.h"

#include <omp.h>
#include <stdlib.h>

void scan(const double *in, double *out, long n)
{
    if (n <= 0) return;

    int nthreads;
    /* Discover the actual thread count inside a parallel region. */
#pragma omp parallel
    {
#pragma omp single
        nthreads = omp_get_num_threads();
    }

    /* block_sum[t] = sum of in[block_start..block_end) for thread t.
     * block_off[t] = exclusive prefix sum of block_sum, i.e. the running
     *                total that thread t must add to every element it writes. */
    double *block_sum = (double *)malloc(sizeof(double) * (size_t)nthreads);
    double *block_off = (double *)malloc(sizeof(double) * (size_t)nthreads);

    /* ------------------------------------------------------------------ */
    /* Pass 1: each thread sums its slice.                                 */
    /* ------------------------------------------------------------------ */
#pragma omp parallel num_threads(nthreads)
    {
        int tid   = omp_get_thread_num();
        int nt    = omp_get_num_threads();

        /* Divide [0, n) into nt contiguous blocks as evenly as possible. */
        long block = (n + nt - 1) / nt;          /* ceiling division      */
        long start = tid * block;
        long end   = start + block;
        if (end > n) end = n;

        double s = 0.0;
        for (long i = start; i < end; i++) s += in[i];
        block_sum[tid] = s;
    }

    /* ------------------------------------------------------------------ */
    /* Serial exclusive scan of block_sum → block_off.                    */
    /* ------------------------------------------------------------------ */
    {
        double running = 0.0;
        for (int t = 0; t < nthreads; t++) {
            block_off[t] = running;
            running += block_sum[t];
        }
    }

    /* ------------------------------------------------------------------ */
    /* Pass 2: each thread writes its inclusive scan starting from offset. */
    /* ------------------------------------------------------------------ */
#pragma omp parallel num_threads(nthreads)
    {
        int tid   = omp_get_thread_num();
        int nt    = omp_get_num_threads();

        long block = (n + nt - 1) / nt;
        long start = tid * block;
        long end   = start + block;
        if (end > n) end = n;

        double run = block_off[tid];
        for (long i = start; i < end; i++) {
            run    += in[i];
            out[i]  = run;
        }
    }

    free(block_sum);
    free(block_off);
}
