#include "kernel_c.h"
#include "problem.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void kernel_c(const double *in, double *out, long n)
{
    if (n <= 0) return;
    
    int num_threads = omp_get_max_threads();
    
    if (n < num_threads * 2) {
        /* Small array: just do serial prefix sum */
        double run = 0.0;
        for (long i = 0; i < n; i++) {
            run += in[i];
            out[i] = run;
        }
        return;
    }
    
    /* Parallel prefix sum using OpenMP */
    /* Step 1: Divide work among threads and compute local sums */
    long chunk_size = (n + num_threads - 1) / num_threads;
    double *partial_sums = (double *)malloc(num_threads * sizeof(double));
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long start = tid * chunk_size;
        long end = start + chunk_size;
        if (end > n) end = n;
        
        /* Step 2: Compute local prefix sum for this chunk */
        double local_sum = 0.0;
        for (long i = start; i < end; i++) {
            local_sum += in[i];
        }
        partial_sums[tid] = local_sum;
        
        #pragma omp barrier
        
        /* Step 3: Compute offsets (prefix sum of partial sums) */
        double offset = 0.0;
        for (int i = 0; i < tid; i++) {
            offset += partial_sums[i];
        }
        
        /* Step 4: Compute final prefix sum with offset */
        double run = offset;
        for (long i = start; i < end; i++) {
            run += in[i];
            out[i] = run;
        }
    }
    
    free(partial_sums);
}
