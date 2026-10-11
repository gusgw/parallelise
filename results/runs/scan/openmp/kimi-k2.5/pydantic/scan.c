#include "scan.h"
#include <stdlib.h>
#include <omp.h>

void scan(const double *in, double *out, long n)
{
    if (n <= 0) return;

    int num_threads = omp_get_max_threads();
    if (n < num_threads) {
        /* Sequential for small arrays */
        double run = 0.0;
        for (long i = 0; i < n; i++) {
            run += in[i];
            out[i] = run;
        }
        return;
    }

    /* Allocate block sums and offsets */
    double *block_sums = (double *)malloc((size_t)num_threads * sizeof(double));
    double *block_offsets = (double *)malloc((size_t)num_threads * sizeof(double));

    /* Pass 1: Each thread sums its block */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long block_size = n / num_threads;
        long remainder = n % num_threads;
        
        long start = tid * block_size + (tid < remainder ? tid : remainder);
        long end = start + block_size + (tid < remainder ? 1 : 0);
        
        double local_sum = 0.0;
        for (long i = start; i < end; i++) {
            local_sum += in[i];
        }
        block_sums[tid] = local_sum;
    }

    /* Serial exclusive scan of block sums to get offsets */
    double acc = 0.0;
    for (int i = 0; i < num_threads; i++) {
        block_offsets[i] = acc;
        acc += block_sums[i];
    }

    /* Pass 2: Each thread scans its block starting from its offset */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long block_size = n / num_threads;
        long remainder = n % num_threads;
        
        long start = tid * block_size + (tid < remainder ? tid : remainder);
        long end = start + block_size + (tid < remainder ? 1 : 0);
        
        double run = block_offsets[tid];
        for (long i = start; i < end; i++) {
            run += in[i];
            out[i] = run;
        }
    }

    free(block_sums);
    free(block_offsets);
}
