#include <stdlib.h>
#include <omp.h>
#include "scan.h"

void scan(const double *in, double *out, long n) {
    if (n <= 0) return;
    int num_threads = 0;
    double *block_sums = NULL;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        // Allocate block_sums once by a single thread
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            block_sums = (double *)malloc(num_threads * sizeof(double));
        }
        // Compute chunk boundaries
        long chunk = (n + num_threads - 1) / num_threads; // ceil division
        long start = tid * chunk;
        long end = start + chunk;
        if (end > n) end = n;
        double sum = 0.0;
        if (start < n) {
            for (long i = start; i < end; ++i) {
                sum += in[i];
                out[i] = sum; // temporary prefix within block
            }
        }
        // Store block sum (0 if no elements)
        block_sums[tid] = sum;

        #pragma omp barrier
        // Compute offset for this block as sum of previous block sums
        double offset = 0.0;
        for (int i = 0; i < tid; ++i) {
            offset += block_sums[i];
        }
        // Add offset to each element of this block
        if (start < n) {
            for (long i = start; i < end; ++i) {
                out[i] += offset;
            }
        }
    }
    free(block_sums);
}
