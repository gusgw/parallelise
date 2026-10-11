#include "kernel_c.h"
#include "problem.h"
#include <omp.h>
#include <stdlib.h>

void kernel_c(const double *in, double *out, long n)
{
    if (n <= 0) return;

    int num_threads = omp_get_max_threads();
    
    /* Allocate array to store partial sums for each thread */
    double *partial_sums = malloc(sizeof(double) * num_threads);
    if (!partial_sums) return;

    /* Step 1: Each thread computes local prefix sum and stores its total */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long chunk_size = (n + num_threads - 1) / num_threads;
        long start = tid * chunk_size;
        long end = start + chunk_size;
        if (start > n) start = n;
        if (end > n) end = n;

        double local_sum = 0.0;
        for (long i = start; i < end; i++) {
            local_sum += in[i];
            out[i] = local_sum;
        }
        partial_sums[tid] = local_sum;
    }

    /* Step 2: Compute offsets (prefix sum of partial sums) */
    double offset = 0.0;
    for (int i = 0; i < num_threads; i++) {
        double temp = partial_sums[i];
        partial_sums[i] = offset;
        offset += temp;
    }

    /* Step 3: Each thread adds its offset to its elements */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long chunk_size = (n + num_threads - 1) / num_threads;
        long start = tid * chunk_size;
        long end = start + chunk_size;
        if (start > n) start = n;
        if (end > n) end = n;

        double offset = partial_sums[tid];
        for (long i = start; i < end; i++) {
            out[i] += offset;
        }
    }

    free(partial_sums);
}
