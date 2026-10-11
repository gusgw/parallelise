// kernel_c.c — parallel prefix sum implementation using OpenMP

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "kernel_c.h"

void kernel_c(const double *in, double *out, long n)
{
    if (n <= 0) return;
    int max_threads = omp_get_max_threads();
    // Allocate array to hold each thread's total sum
    double *thread_sum = (double *)calloc((size_t)max_threads, sizeof(double));
    if (!thread_sum) {
        fprintf(stderr, "kernel_c: memory allocation failed\n");
        return;
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        // Compute chunk boundaries (balanced distribution)
        long chunk = (n + nthreads - 1) / nthreads; // ceil division
        long start = tid * chunk;
        long end = start + chunk;
        if (end > n) end = n;

        double local_sum = 0.0;
        for (long i = start; i < end; ++i) {
            local_sum += in[i];
            out[i] = local_sum;
        }
        thread_sum[tid] = local_sum;

        #pragma omp barrier
        // Compute offset: sum of all previous thread sums
        double offset = 0.0;
        for (int i = 0; i < tid; ++i) {
            offset += thread_sum[i];
        }
        // Apply offset to our portion of the output (skip if offset is zero)
        if (offset != 0.0) {
            for (long i = start; i < end; ++i) {
                out[i] += offset;
            }
        }
    }

    free(thread_sum);
}
