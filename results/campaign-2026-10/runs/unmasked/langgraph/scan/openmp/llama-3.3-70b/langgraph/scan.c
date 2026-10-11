#include "scan.h"
#include <omp.h>
#include <stdlib.h>

void scan(const double *in, double *out, long n) {
    long block_size = n / omp_get_max_threads();
    long remainder = n % omp_get_max_threads();
    long start, end;
    double *block_sums = (double *) malloc(omp_get_max_threads() * sizeof(double));
    #pragma omp parallel
    {
        long tid = omp_get_thread_num();
        start = tid * block_size + (tid < remainder ? tid : remainder);
        end = (tid + 1) * block_size + (tid + 1 < remainder ? tid + 1 : remainder);
        double sum = 0;
        for (long i = start; i < end; i++) {
            sum += in[i];
        }
        block_sums[tid] = sum;
    }
    double exclusive_sum = 0;
    for (long i = 0; i < omp_get_max_threads(); i++) {
        double temp = block_sums[i];
        block_sums[i] = exclusive_sum;
        exclusive_sum += temp;
    }
    #pragma omp parallel
    {
        long tid = omp_get_thread_num();
        start = tid * block_size + (tid < remainder ? tid : remainder);
        end = (tid + 1) * block_size + (tid + 1 < remainder ? tid + 1 : remainder);
        double block_offset = block_sums[tid];
        double sum = block_offset;
        for (long i = start; i < end; i++) {
            out[i] = sum;
            sum += in[i];
        }
    }
    free(block_sums);
}
