#include "scan.h"
#include <omp.h>
#include <stdlib.h>
#include <string.h>

void scan(const double *in, double *out, long n) {
    if (n <= 0) return;

    int num_threads = omp_get_max_threads();
    if (num_threads <= 1 || n < 1000) {
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
        long n_threads = omp_get_num_threads();
        long block_size = (n + n_threads - 1) / n_threads;
        long start = tid * block_size;
        long end = (start + block_size > n) ? n : start + block_size;

        double local_sum = 0.0;
        for (long i = start; i < end; i++) {
            local_sum += in[i];
        }
        block_sums[tid] = local_sum;

        #pragma omp barrier

        // Only one thread computes the prefix sum of block sums
        #pragma omp single
        {
            double run = 0.0;
            for (int i = 0; i < n_threads; i++) {
                double temp = block_sums[i];
                run += temp;
                block_sums[i] = run - temp; // Convert to exclusive scan
            }
        }

        #pragma omp barrier

        double offset = block_sums[tid];
        double run = offset;
        for (long i = start; i < end; i++) {
            run += in[i];
            out[i] = run;
        }
    }

    free(block_sums);
}
