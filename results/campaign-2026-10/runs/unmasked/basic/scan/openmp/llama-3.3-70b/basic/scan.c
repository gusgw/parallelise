#include "scan.h"
#include <omp.h>
#include <stdlib.h>

void scan(const double *in, double *out, long n) {
    long block_size = n / omp_get_max_threads();
    double *block_sums = (double *) malloc(omp_get_max_threads() * sizeof(double));
    #pragma omp parallel
    {
        long tid = omp_get_thread_num();
        long start = tid * block_size;
        long end = (tid + 1) * block_size;
        if (tid == omp_get_max_threads() - 1) end = n;
        double block_sum = 0;
        for (long i = start; i < end; i++) {
            block_sum += in[i];
        }
        block_sums[tid] = block_sum;
    }
    double prefix_sum = 0;
    for (long i = 0; i < omp_get_max_threads(); i++) {
        double new_sum = prefix_sum;
        #pragma omp parallel for reduction(+:new_sum)
        for (long j = i * block_size; j < (i + 1) * block_size; j++) {
            if (j < n) {
                new_sum += in[j];
            }
        }
        #pragma omp parallel for
        for (long j = i * block_size; j < (i + 1) * block_size; j++) {
            if (j < n) {
                out[j] = prefix_sum + in[j];
            }
        }
        prefix_sum += block_sums[i];
    }
    #pragma omp parallel for
    for (long j = 1; j < n; j++) {
        out[j] += out[j-1];
    }
    free(block_sums);
}
