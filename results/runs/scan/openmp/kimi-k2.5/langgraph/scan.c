#include "scan.h"
#include <stdlib.h>
#include <omp.h>

void scan(const double *in, double *out, long n)
{
    if (n <= 0) return;

    int num_threads = omp_get_max_threads();
    if (num_threads < 1) num_threads = 1;

    /* Compute block size for each thread */
    long block_size = (n + num_threads - 1) / num_threads;
    if (block_size < 1) block_size = 1;

    /* Allocate array for block sums */
    double *block_sums = (double *)malloc((size_t)num_threads * sizeof(double));
    if (!block_sums) return;

    /* Pass 1: Each thread computes the sum of its block */
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < num_threads; t++) {
        long start = (long)t * block_size;
        long end = start + block_size;
        if (start > n) start = n;
        if (end > n) end = n;

        double sum = 0.0;
        for (long i = start; i < end; i++) {
            sum += in[i];
        }
        block_sums[t] = sum;
    }

    /* Serial exclusive scan on block sums to get offsets */
    double running = 0.0;
    for (int t = 0; t < num_threads; t++) {
        double temp = block_sums[t];
        block_sums[t] = running;
        running += temp;
    }

    /* Pass 2: Each thread scans its block starting from its offset */
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < num_threads; t++) {
        long start = (long)t * block_size;
        long end = start + block_size;
        if (start > n) start = n;
        if (end > n) end = n;

        double run = block_sums[t];
        for (long i = start; i < end; i++) {
            run += in[i];
            out[i] = run;
        }
    }

    free(block_sums);
}
