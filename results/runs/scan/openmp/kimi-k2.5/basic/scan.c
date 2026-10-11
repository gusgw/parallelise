#include "scan.h"
#include <stdlib.h>
#include <omp.h>

/* Inclusive prefix sum using two-pass block scan:
 * Pass 1: Each thread sums its block of input
 * Serial exclusive scan of block sums gives each block's starting offset
 * Pass 2: Each thread scans its block, starting from its offset
 */
void scan(const double *in, double *out, long n)
{
    if (n <= 0) return;
    
    int num_threads = omp_get_max_threads();
    if (num_threads < 1) num_threads = 1;
    
    /* Allocate array for block sums (one per thread) */
    double *block_sums = (double *)malloc(num_threads * sizeof(double));
    if (!block_sums) return;
    
    /* Pass 1: Each thread computes sum of its block */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long block_size = (n + num_threads - 1) / num_threads;
        long start = tid * block_size;
        long end = start + block_size;
        if (end > n) end = n;
        
        double local_sum = 0.0;
        for (long i = start; i < end; i++) {
            local_sum += in[i];
        }
        block_sums[tid] = local_sum;
    }
    
    /* Serial exclusive scan of block sums to get offsets */
    double running = 0.0;
    for (int i = 0; i < num_threads; i++) {
        double temp = block_sums[i];
        block_sums[i] = running;
        running += temp;
    }
    
    /* Pass 2: Each thread scans its block, starting from its offset */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long block_size = (n + num_threads - 1) / num_threads;
        long start = tid * block_size;
        long end = start + block_size;
        if (end > n) end = n;
        
        double local_sum = block_sums[tid];
        for (long i = start; i < end; i++) {
            local_sum += in[i];
            out[i] = local_sum;
        }
    }
    
    free(block_sums);
}
