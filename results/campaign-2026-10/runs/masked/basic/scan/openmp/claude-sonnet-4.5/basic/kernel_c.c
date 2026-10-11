#include "kernel_c.h"
#include <omp.h>

void kernel_c(const double * restrict in, double * restrict out, long n)
{
    static double chunk_sums[256 * 16];
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        long chunk_size = n / num_threads;
        long start = tid * chunk_size;
        long end = (tid == num_threads - 1) ? n : start + chunk_size;
        
        // Phase 1: Local prefix sum
        double run = 0.0;
        for (long i = start; i < end; i++) {
            run += in[i];
            out[i] = run;
        }
        chunk_sums[tid * 16] = run;
        
        #pragma omp barrier
        
        // Phase 2: Prefix sum of chunk sums (single thread)
        #pragma omp master
        {
            double offset = 0.0;
            for (int j = 0; j < num_threads; j++) {
                double temp = chunk_sums[j * 16];
                chunk_sums[j * 16] = offset;
                offset += temp;
            }
        }
        
        #pragma omp barrier
        
        // Phase 3: Add offset
        if (tid > 0) {
            double offset_val = chunk_sums[tid * 16];
            for (long i = start; i < end; i++) {
                out[i] += offset_val;
            }
        }
    }
}
