/* histogram.c — parallel OpenMP histogram kernel.
 *
 * Strategy: each thread builds a PRIVATE local histogram over its share of the
 * input, then merges into the shared counts[] array with an atomic add.  This
 * avoids any data race on the hot loop and keeps the merge cost O(nbins * nthreads),
 * which is negligible compared with the O(n) binning work. */

#include <stdlib.h>
#include <string.h>

#include "histogram.h"

void histogram(const double *x, long n, long *counts, int nbins)
{
    /* Zero the output array (single-threaded; caller may not have done it). */
    memset(counts, 0, sizeof(long) * (size_t)nbins);

#pragma omp parallel
    {
        /* Each thread allocates its own private histogram and zeroes it. */
        long *local = (long *)calloc((size_t)nbins, sizeof(long));
        if (local) {
            /* Bin every element that belongs to this thread's slice. */
#pragma omp for schedule(static)
            for (long i = 0; i < n; i++) {
                int b = (int)(x[i] * nbins);
                if (b >= nbins) b = nbins - 1;
                local[b]++;
            }

            /* Merge local histogram into the shared output with atomic adds so
             * that threads writing different bins never race. */
            for (int b = 0; b < nbins; b++) {
                if (local[b] != 0) {
#pragma omp atomic
                    counts[b] += local[b];
                }
            }

            free(local);
        }
    }
}
