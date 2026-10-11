#include "histogram.h"

#include <omp.h>
#include <stdlib.h>
#include <string.h>

void histogram(const double *x, long n, long *counts, int nbins)
{
    /* Zero the output counts */
    for (int b = 0; b < nbins; b++) {
        counts[b] = 0;
    }

    /* Determine number of threads for local histogram allocation */
    int nthreads = omp_get_max_threads();
    
    /* Allocate a 2D array: nthreads x nbins, flattened */
    long *local_counts = NULL;
    if (nthreads > 1) {
        local_counts = (long *)calloc((size_t)nthreads * nbins, sizeof(long));
    }

    /* Parallel histogram computation */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long *my_counts = (nthreads > 1) ? &local_counts[(size_t)tid * nbins] : counts;

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
    if (nthreads > 1 && local_counts != NULL) {
        for (int t = 0; t < nthreads; t++) {
            for (int b = 0; b < nbins; b++) {
                counts[b] += local_counts[(size_t)t * nbins + b];
            }
        }
        free(local_counts);
    }
}
