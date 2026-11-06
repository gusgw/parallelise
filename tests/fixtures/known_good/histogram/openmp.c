#include <stdlib.h>

#include "histogram.h"

/* Reference solution (validates the harness -- NOT shown to the agents). Each thread
 * bins its share of the values into a PRIVATE local histogram -- no shared writes in
 * the hot loop, so there is nothing to race on -- and then merges its local histogram
 * into the shared counts inside a critical section. The merge is nbins additions per
 * thread, so the serialisation cost is negligible. */
void histogram(const double *x, long n, long *counts, int nbins)
{
    for (int b = 0; b < nbins; b++) counts[b] = 0;

    #pragma omp parallel
    {
        long *local = calloc((size_t)nbins, sizeof(long));
        #pragma omp for nowait
        for (long i = 0; i < n; i++) {
            int b = (int)(x[i] * (double)nbins);
            if (b >= nbins) b = nbins - 1;
            local[b]++;
        }
        #pragma omp critical
        for (int b = 0; b < nbins; b++) counts[b] += local[b];
        free(local);
    }
}
