#include "histogram.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void histogram(const double *x, long n, long *counts, int nbins)
{
    /* Zero the shared counts array */
    for (int b = 0; b < nbins; b++) {
        counts[b] = 0;
    }

    /* Get number of threads for allocating local histograms */
    int nthreads = omp_get_max_threads();

    /* Allocate local histograms for each thread (flattened 2D array) */
    long *local_counts = calloc((size_t)nthreads * nbins, sizeof(long));
    if (local_counts == NULL) {
        return;  /* Error: allocation failed */
    }

    /* Parallel histogram computation with private local bins */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long *my_counts = &local_counts[(size_t)tid * nbins];

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
    for (int t = 0; t < nthreads; t++) {
        long *thread_counts = &local_counts[(size_t)t * nbins];
        for (int b = 0; b < nbins; b++) {
            counts[b] += thread_counts[b];
        }
    }

    free(local_counts);
}
