// scan.c – parallel inclusive prefix sum using OpenMP block scan

#include <omp.h>
#include <stdlib.h>
#include <stddef.h>

#include "scan.h"

void scan(const double *in, double *out, long n)
{
    if (n <= 0) return;
    int num_threads = omp_get_max_threads();
    // Allocate temporary array for each thread's block sum
    double *block_sum = (double *)malloc(sizeof(double) * num_threads);
    if (!block_sum) return; // allocation failure, silently return (should not happen)

    // First pass: each thread computes sum of its block
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long chunk = (n + num_threads - 1) / num_threads; // ceil division
        long start = tid * chunk;
        long end = start + chunk;
        if (end > n) end = n;
        double sum = 0.0;
        for (long i = start; i < end; ++i) {
            sum += in[i];
        }
        block_sum[tid] = sum;
    }

    // Compute exclusive scan of block sums to obtain offsets for each block
    double *offset = (double *)malloc(sizeof(double) * num_threads);
    if (!offset) {
        free(block_sum);
        return;
    }
    offset[0] = 0.0;
    for (int i = 1; i < num_threads; ++i) {
        offset[i] = offset[i-1] + block_sum[i-1];
    }

    // Second pass: each thread scans its own block, adding the block offset
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long chunk = (n + num_threads - 1) / num_threads;
        long start = tid * chunk;
        long end = start + chunk;
        if (end > n) end = n;
        double run = offset[tid];
        for (long i = start; i < end; ++i) {
            run += in[i];
            out[i] = run;
        }
    }

    free(offset);
    free(block_sum);
}
