#include <stdlib.h>
#include <omp.h>
#include "scan.h"

void scan(const double *in, double *out, long n)
{
    if (n <= 0) return;

    int num_threads = omp_get_max_threads();
    double *block_sums = (double *)malloc(num_threads * sizeof(double));
    double *offsets   = (double *)malloc(num_threads * sizeof(double));

    /* First pass: each thread computes the sum of its block */
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        long chunk = (n + num_threads - 1) / num_threads; // ceil division
        long start = tid * chunk;
        long end   = start + chunk;
        if (end > n) end = n;
        double sum = 0.0;
        for (long i = start; i < end; ++i) {
            sum += in[i];
        }
        block_sums[tid] = sum;
    }

    /* Exclusive scan of block sums to obtain offsets */
    offsets[0] = 0.0;
    for (int i = 1; i < num_threads; ++i) {
        offsets[i] = offsets[i-1] + block_sums[i-1];
    }

    /* Second pass: each thread scans its block, adding the offset */
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        long chunk = (n + num_threads - 1) / num_threads;
        long start = tid * chunk;
        long end   = start + chunk;
        if (end > n) end = n;
        double run = offsets[tid];
        for (long i = start; i < end; ++i) {
            run += in[i];
            out[i] = run;
        }
    }

    free(block_sums);
    free(offsets);
}
