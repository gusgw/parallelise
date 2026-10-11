#include <stdlib.h>
#include <omp.h>
#include "kernel_c.h"

void kernel_c(const double *in, double *out, long n)
{
    if (n <= 0) return;

    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    // Allocate array to hold the total sum of each thread's chunk.
    double *thread_sum = (double *)malloc((num_threads + 1) * sizeof(double));
    if (!thread_sum) return; // allocation failure, nothing to do
    thread_sum[0] = 0.0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long start = (long)tid * n / num_threads;
        long end   = (long)(tid + 1) * n / num_threads; // exclusive
        double sum = 0.0;
        for (long i = start; i < end; ++i) {
            sum += in[i];
            out[i] = sum;          // local prefix sum within the chunk
        }
        thread_sum[tid + 1] = sum; // store the chunk total

        #pragma omp barrier
        // Compute the exclusive prefix of the chunk totals once.
        #pragma omp single
        {
            for (int i = 1; i <= num_threads; ++i) {
                thread_sum[i] += thread_sum[i - 1];
            }
        }
        #pragma omp barrier

        double offset = thread_sum[tid];
        for (long i = start; i < end; ++i) {
            out[i] += offset;      // add the sum of all previous chunks
        }
    }

    free(thread_sum);
}
