/* scan.c — parallel inclusive prefix sum using a two-pass block scan.
 *
 * Algorithm (matches the description in scan.h):
 *
 *   Pass 1 – each OpenMP thread independently sums its contiguous slice of
 *             `in[]` and stores the result in block_sum[tid].
 *
 *   Serial bridge – an exclusive prefix scan over block_sum[] gives each
 *             thread its starting offset (the sum of all elements that
 *             belong to earlier threads).
 *
 *   Pass 2 – each thread walks its slice again, this time accumulating from
 *             its offset and writing the inclusive running total into out[].
 *
 * This is the textbook shared-memory block scan; every element of `in` is
 * read exactly twice and every element of `out` is written exactly once.
 * No thread repeats work that belongs to another thread.
 */

#include <stdlib.h>   /* malloc, free */
#include <omp.h>

#include "scan.h"

void scan(const double *in, double *out, long n)
{
    if (n <= 0) return;

    /* ------------------------------------------------------------------ */
    /* Determine the number of threads that will actually be used.         */
    /* We query inside a parallel region so the runtime can honour any     */
    /* OMP_NUM_THREADS / omp_set_num_threads() setting in effect.          */
    /* ------------------------------------------------------------------ */
    int nthreads;
#pragma omp parallel
    {
#pragma omp single
        nthreads = omp_get_num_threads();
    }

    /* Per-block partial sums (one entry per thread). */
    double *block_sum = (double *)malloc(sizeof(double) * (size_t)nthreads);
    if (!block_sum) return;   /* allocation failure – nothing we can do */

    /* ================================================================== */
    /* Pass 1: each thread sums its own slice of in[].                     */
    /* ================================================================== */
#pragma omp parallel num_threads(nthreads)
    {
        int tid   = omp_get_thread_num();
        int nt    = omp_get_num_threads();

        /* Divide [0, n) into nt contiguous blocks as evenly as possible. */
        long block = (n + nt - 1) / nt;   /* ceiling division            */
        long lo    = tid * block;
        long hi    = lo + block;
        if (hi > n) hi = n;

        double s = 0.0;
        for (long i = lo; i < hi; i++)
            s += in[i];

        block_sum[tid] = s;
    } /* implicit barrier – all block_sum[] values are visible below */

    /* ================================================================== */
    /* Serial bridge: exclusive prefix scan of block_sum[].               */
    /* block_sum[tid] becomes the sum of all elements BEFORE thread tid.  */
    /* ================================================================== */
    {
        double running = 0.0;
        for (int t = 0; t < nthreads; t++) {
            double tmp    = block_sum[t];
            block_sum[t]  = running;   /* exclusive: offset for thread t */
            running      += tmp;
        }
    }

    /* ================================================================== */
    /* Pass 2: each thread writes its slice of out[] starting from its    */
    /* offset (block_sum[tid]).                                            */
    /* ================================================================== */
#pragma omp parallel num_threads(nthreads)
    {
        int tid   = omp_get_thread_num();
        int nt    = omp_get_num_threads();

        long block = (n + nt - 1) / nt;
        long lo    = tid * block;
        long hi    = lo + block;
        if (hi > n) hi = n;

        double acc = block_sum[tid];   /* start from this thread's offset */
        for (long i = lo; i < hi; i++) {
            acc   += in[i];
            out[i] = acc;
        }
    }

    free(block_sum);
}
