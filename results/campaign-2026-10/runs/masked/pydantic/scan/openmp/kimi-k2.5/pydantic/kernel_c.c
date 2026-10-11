#include "kernel_c.h"
#include <stdlib.h>
#include <omp.h>

void kernel_c(const double *in, double *out, long n)
{
    if (n <= 0) return;
    
    /* Parallel prefix sum using OpenMP */
    /* Use a block-based approach with minimal overhead */
    
    #pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        
        /* Calculate block boundaries */
        long block_size = n / num_threads;
        long remainder = n % num_threads;
        
        long start = tid * block_size + (tid < remainder ? tid : remainder);
        long end = start + block_size + (tid < remainder ? 1 : 0);
        
        /* Step 1: Local prefix sum within each block */
        double local_sum = 0.0;
        for (long i = start; i < end; i++) {
            local_sum += in[i];
            out[i] = local_sum;
        }
        
        /* Step 2: Compute offsets using a shared array */
        /* Use a simple barrier-based approach */
        static double *offsets = NULL;
        #pragma omp single
        {
            offsets = (double *)malloc(num_threads * sizeof(double));
        }
        
        offsets[tid] = local_sum;
        
        #pragma omp barrier
        
        /* Step 3: Prefix sum of offsets (only one thread does this) */
        #pragma omp single
        {
            double running = 0.0;
            for (int i = 0; i < num_threads; i++) {
                double temp = offsets[i];
                offsets[i] = running;
                running += temp;
            }
        }
        
        /* Step 4: Add offset to each element in the block */
        double offset = offsets[tid];
        for (long i = start; i < end; i++) {
            out[i] += offset;
        }
        
        #pragma omp single
        {
            free(offsets);
            offsets = NULL;
        }
    }
}
