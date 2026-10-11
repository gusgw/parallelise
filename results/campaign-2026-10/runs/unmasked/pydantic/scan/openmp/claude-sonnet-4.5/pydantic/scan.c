#include "scan.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/* Inclusive prefix sum of a length-n array:
 *     out[i] = in[0] + in[1] + ... + in[i]   for 0 <= i < n.
 *
 * Parallelise with the classic two-pass block scan:
 * Pass 1: each thread sums its contiguous block of in
 * Serial exclusive scan of the block sums gives each block's starting offset
 * Pass 2: each thread scans its own block, starting from its offset
 */
void scan(const double *in, double *out, long n)
{
    if (n <= 0) return;

    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    /* Allocate block sums for each thread */
    double *block_sums = (double *)malloc(num_threads * sizeof(double));
    
    /* Pass 1: each thread computes the sum of its block */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long block_size = (n + num_threads - 1) / num_threads;
        long start = tid * block_size;
        long end = start + block_size;
        if (end > n) end = n;
        
        double sum = 0.0;
        for (long i = start; i < end; i++) {
            sum += in[i];
        }
        block_sums[tid] = sum;
    }
    
    /* Serial exclusive scan of block sums to get offsets */
    double offset = 0.0;
    for (int t = 0; t < num_threads; t++) {
        double tmp = block_sums[t];
        block_sums[t] = offset;
        offset += tmp;
    }
    
    /* Pass 2: each thread does inclusive scan of its block with its offset */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long block_size = (n + num_threads - 1) / num_threads;
        long start = tid * block_size;
        long end = start + block_size;
        if (end > n) end = n;
        
        double run = block_sums[tid];
        for (long i = start; i < end; i++) {
            run += in[i];
            out[i] = run;
        }
    }
    
    free(block_sums);
}
