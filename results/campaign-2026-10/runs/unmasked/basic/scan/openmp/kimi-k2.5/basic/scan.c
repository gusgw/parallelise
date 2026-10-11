#include "scan.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void scan(const double *in, double *out, long n)
{
    if (n <= 0) return;
    
    int num_threads = omp_get_max_threads();
    if (num_threads > n) num_threads = (int)n;
    if (num_threads < 1) num_threads = 1;
    
    /* Allocate array for block sums */
    double *block_sums = (double *)malloc((size_t)num_threads * sizeof(double));
    if (!block_sums) return;
    
    /* Calculate block size for each thread */
    long block_size = n / num_threads;
    long remainder = n % num_threads;
    
    /* Pass 1: Each thread sums its block */
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        
        /* Compute start and end indices for this thread's block */
        long start = tid * block_size + (tid < remainder ? tid : remainder);
        long end = start + block_size + (tid < remainder ? 1 : 0);
        
        /* Compute local sum */
        double local_sum = 0.0;
        for (long i = start; i < end; i++) {
            local_sum += in[i];
        }
        block_sums[tid] = local_sum;
    }
    
    /* Exclusive scan of block sums (serial) */
    double running = 0.0;
    for (int i = 0; i < num_threads; i++) {
        double temp = block_sums[i];
        block_sums[i] = running;
        running += temp;
    }
    
    /* Pass 2: Each thread scans its block, starting from its offset */
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        
        /* Compute start and end indices for this thread's block */
        long start = tid * block_size + (tid < remainder ? tid : remainder);
        long end = start + block_size + (tid < remainder ? 1 : 0);
        
        /* Scan this block starting from the block's offset */
        double acc = block_sums[tid];
        for (long i = start; i < end; i++) {
            acc += in[i];
            out[i] = acc;
        }
    }
    
    free(block_sums);
}
