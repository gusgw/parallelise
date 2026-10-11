#include "histogram.h"

#include <omp.h>
#include <string.h>

void histogram(const double *x, long n, long *counts, int nbins)
{
    /* Zero the output counts */
    for (int b = 0; b < nbins; b++) {
        counts[b] = 0;
    }

    /* Each thread gets a private histogram to avoid data races */
    int nthreads = omp_get_max_threads();
    long *local_counts = NULL;
    if (nthreads > 1) {
        local_counts = __builtin_alloca(sizeof(long) * nbins * nthreads);
        memset(local_counts, 0, sizeof(long) * nbins * nthreads);
    }

    /* Parallel histogram computation */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long *my_counts = (nthreads > 1) ? &local_counts[tid * nbins] : counts;

        #pragma omp for
        for (long i = 0; i < n; i++) {
            int b = (int)(x[i] * nbins);
            if (b >= nbins) {
                b = nbins - 1;
            }
            my_counts[b]++;
        }
    }

    /* Merge local histograms into shared counts */
    if (nthreads > 1) {
        for (int t = 0; t < nthreads; t++) {
            for (int b = 0; b < nbins; b++) {
                counts[b] += local_counts[t * nbins + b];
            }
        }
    }
}
