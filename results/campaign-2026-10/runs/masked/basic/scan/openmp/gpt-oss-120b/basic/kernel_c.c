#include <stdlib.h>
#include <omp.h>
#include "kernel_c.h"

void kernel_c(const double *in, double *out, long n) {
    if (n <= 0) return;
    int num_threads = 0;
    // Determine number of threads used in parallel region
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    double *thread_sum = (double*)malloc(num_threads * sizeof(double));
    if (!thread_sum) return; // allocation failure, silently return

    // First pass: compute local prefix sums and store total sum per thread
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long chunk = (n + num_threads - 1) / num_threads; // ceiling division
        long start = tid * chunk;
        long end = start + chunk;
        if (end > n) end = n;
        double run = 0.0;
        for (long i = start; i < end; ++i) {
            run += in[i];
            out[i] = run;
        }
        thread_sum[tid] = run;
    }

    // Compute offsets for each thread (exclusive prefix sum of thread_sum)
    double *offset = (double*)malloc(num_threads * sizeof(double));
    if (!offset) {
        free(thread_sum);
        return;
    }
    offset[0] = 0.0;
    for (int i = 1; i < num_threads; ++i) {
        offset[i] = offset[i-1] + thread_sum[i-1];
    }

    // Second pass: add offset to each element in the thread's segment
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long chunk = (n + num_threads - 1) / num_threads;
        long start = tid * chunk;
        long end = start + chunk;
        if (end > n) end = n;
        double off = offset[tid];
        for (long i = start; i < end; ++i) {
            out[i] += off;
        }
    }

    free(thread_sum);
    free(offset);
}
