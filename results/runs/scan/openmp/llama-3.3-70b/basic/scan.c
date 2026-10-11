#include "scan.h"
#include <stdlib.h>
#include <omp.h>

void scan(const double *in, double *out, long n) {
    long block_size = 1000;
    long num_blocks = n / block_size;
    double *block_sums = (double *) malloc(num_blocks * sizeof(double));
    #pragma omp parallel
    {
        long tid = omp_get_thread_num();
        long start = tid * block_size;
        long end = (tid + 1) * block_size;
        if (end > n) end = n;
        double block_sum = 0.0;
        for (long i = start; i < end; i++) {
            block_sum += in[i];
        }
        block_sums[tid] = block_sum;
    }
    double exclusive_sum = 0.0;
    for (long i = 0; i < num_blocks; i++) {
        double block_sum = block_sums[i];
        block_sums[i] = exclusive_sum;
        exclusive_sum += block_sum;
    }
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        long block_idx = i / block_size;
        double block_offset = block_sums[block_idx];
        double prefix_sum = block_offset;
        for (long j = block_idx * block_size; j <= i; j++) {
            prefix_sum += in[j];
        }
        out[i] = prefix_sum;
    }
    free(block_sums);
}
