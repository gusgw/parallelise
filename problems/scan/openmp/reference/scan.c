#include <omp.h>
#include <stdlib.h>

#include "scan.h"

/* Reference solution (validates the harness -- NOT shown to the agents). The classic
 * two-pass block scan: each thread owns one contiguous block. Pass 1 sums each block;
 * a serial exclusive scan of the block sums (cheap: one value per thread) gives each
 * block's starting offset; pass 2 scans each block from its offset. One parallel
 * region; the barrier separates the passes. */
void scan(const double *in, double *out, long n)
{
    if (n <= 0)
        return;
    double *block_sums = NULL;
    #pragma omp parallel
    {
        const int nt = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        #pragma omp single
        block_sums = malloc(sizeof(double) * (size_t)nt);
        /* the single's implicit barrier makes block_sums visible to every thread */

        const long lo = n * tid / nt;
        const long hi = n * (tid + 1) / nt;

        /* pass 1: each thread sums its own block */
        double s = 0.0;
        for (long i = lo; i < hi; i++) s += in[i];
        block_sums[tid] = s;
        #pragma omp barrier

        /* serial exclusive scan of the block sums -> each block's starting offset */
        #pragma omp single
        {
            double run = 0.0;
            for (int t = 0; t < nt; t++) {
                double b = block_sums[t];
                block_sums[t] = run;
                run += b;
            }
        }
        /* the single's implicit barrier orders the offsets before pass 2 */

        /* pass 2: each thread scans its own block, starting from its offset */
        double run = block_sums[tid];
        for (long i = lo; i < hi; i++) {
            run += in[i];
            out[i] = run;
        }
    }
    free(block_sums);
}
