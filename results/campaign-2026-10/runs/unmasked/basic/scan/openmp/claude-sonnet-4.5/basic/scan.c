#include "scan.h"
#include <stdlib.h>
#include <omp.h>

/* Inclusive prefix sum of a length-n array:
 *     out[i] = in[0] + in[1] + ... + in[i]   for 0 <= i < n.
 *
 * Two-pass block scan: pass 1, each thread sums its contiguous block of in;
 * a serial exclusive scan of the block sums gives each block's starting offset;
 * pass 2, each thread scans its own block, starting from its offset. */
void scan(const double *in, double *out, long n)
{
    int nthreads;
    double *block_sums;
    
    /* Determine number of threads */
    #pragma omp parallel
    {
        #pragma omp single
        {
            nthreads = omp_get_num_threads();
        }
    }
    
    /* Allocate array for block sums */
    block_sums = (double *)malloc(nthreads * sizeof(double));
    
    /* Pass 1: Each thread computes the sum of its block */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long block_size = (n + nthreads - 1) / nthreads;
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
    for (int t = 0; t < nthreads; t++) {
        double tmp = block_sums[t];
        block_sums[t] = offset;
        offset += tmp;
    }
    
    /* Pass 2: Each thread scans its own block with its offset */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long block_size = (n + nthreads - 1) / nthreads;
        long start = tid * block_size;
        long end = start + block_size;
        if (end > n) end = n;
        
        double running = block_sums[tid];
        for (long i = start; i < end; i++) {
            running += in[i];
            out[i] = running;
        }
    }
    
    free(block_sums);
}
