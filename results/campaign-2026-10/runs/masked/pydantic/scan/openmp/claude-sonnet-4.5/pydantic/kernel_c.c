#include "kernel_c.h"
#include <omp.h>
#include <stdlib.h>

void kernel_c(const double *in, double *out, long n)
{
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    
    double *chunk_sums = malloc(sizeof(double) * num_threads);
    if (!chunk_sums) return;
    
    // Phase 1: Compute local cumulative sums within each chunk
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long chunk_size = (n + num_threads - 1) / num_threads;
        long start = tid * chunk_size;
        long end = start + chunk_size;
        if (end > n) end = n;
        
        double run = 0.0;
        if (start < n) {
            for (long i = start; i < end; i++) {
                run += in[i];
                out[i] = run;
            }
        }
        chunk_sums[tid] = run;
    }
    
    // Phase 2: Compute prefix sum of chunk sums (serial)
    double offset = 0.0;
    for (int t = 0; t < num_threads; t++) {
        double temp = chunk_sums[t];
        chunk_sums[t] = offset;
        offset += temp;
    }
    
    // Phase 3: Add offsets to each chunk
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long chunk_size = (n + num_threads - 1) / num_threads;
        long start = tid * chunk_size;
        long end = start + chunk_size;
        if (end > n) end = n;
        
        double chunk_offset = chunk_sums[tid];
        for (long i = start; i < end; i++) {
            out[i] += chunk_offset;
        }
    }
    
    free(chunk_sums);
}
