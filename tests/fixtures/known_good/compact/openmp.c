/* My own OpenMP routine for compact, written only to test the notebook; not a known-good routine. */
#include <omp.h>
#include <stdlib.h>

#include "compact.h"

void compact(const double *x, double threshold, long *indices, long *indices_count, long n)
{
    long *starts = NULL;
    long total = 0;
#pragma omp parallel
    {
        int threads = omp_get_num_threads();
        int me = omp_get_thread_num();
#pragma omp single
        starts = calloc((size_t)threads + 1, sizeof(long));
        long lo = n * me / threads;
        long hi = n * (me + 1) / threads;
        long found = 0;
        for (long i = lo; i < hi; i++) {
            if (x[i] > threshold) {
                found++;
            }
        }
        starts[me + 1] = found;
#pragma omp barrier
#pragma omp single
        {
            for (int t = 0; t < threads; t++) {
                starts[t + 1] += starts[t];
            }
            total = starts[threads];
        }
        long place = starts[me];
        for (long i = lo; i < hi; i++) {
            if (x[i] > threshold) {
                indices[place++] = i;
            }
        }
    }
    free(starts);
    *indices_count = total;
}
