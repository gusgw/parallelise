#include "scan.h"
#include <omp.h>
#include <stdlib.h>

void scan(const double *in, double *out, long n) {
    if (n <= 0) return;

    int num_threads = omp_get_max_threads();
    if (num_threads <= 1 || n < 1024) {
        double run = 0.0;
        for (long i = 0; i < n; i++) {
            run += in[i];
            out[i] = run;
        }
        return;
    }

    double *block_sums = (double *)malloc(num_threads * sizeof(double));
    if (!block_sums) return;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long n_per_thread = n / num_threads;
        long start = tid * n_per_thread;
        long end = (tid == num_threads - 1) ? n : (tid + 1) * n_per_thread;

        double local_sum = 0.0;
        for (long i = start; i < end; i++) {
            local_sum += in[i];
        }
        block_sums[tid] = local_sum;

        #pragma omp barrier

        // Only thread 0 computes the prefix sum of block sums
        #pragma omp single
        {
            double run = 0.0;
            for (int i = 0; i < num_threads; i++) {
                double temp = block_sums[i];
                run += temp;
                block_sums[i] = run;
            }
            // For inclusive scan, the first block starts at 0, 
            // and subsequent blocks start at the sum of previous blocks.
            // block_sums[i] currently holds the inclusive sum.
            // We need the exclusive sum for the offsets.
            double prev = 0.0;
            for (int i = 0; i < num_threads; i++) {
                double current = block_sums[i];
                block_sums[i] = prev;
                prev = current;
            }
        }

        double offset = block_sums[tid];
        double run = offset;
        for (long i = start; i < end; i++) {
            run += in[i];
            out[i] = run;
        }
    }

    free(block_sums);
}
