/* histogram.c — parallel OpenMP histogram kernel.
 *
 * Strategy: each thread builds a PRIVATE local histogram over its share of the
 * input array, then merges into the shared counts[] array with an atomic add.
 * This avoids any data race on the shared bins while keeping the hot loop
 * entirely thread-local (no synchronisation inside the loop).
 */

#include <stdlib.h>
#include <string.h>

#include "histogram.h"

void histogram(const double *x, long n, long *counts, int nbins)
{
    /* Zero the output histogram. */
    memset(counts, 0, sizeof(long) * (size_t)nbins);

#pragma omp parallel
    {
        /* Each thread allocates its own private histogram on the stack (nbins is
         * small — 256 by default) and zeroes it. */
        long *local = (long *)calloc((size_t)nbins, sizeof(long));
        if (local) {
            /* Partition the input across threads. */
#pragma omp for schedule(static)
            for (long i = 0; i < n; i++) {
                int b = (int)(x[i] * (double)nbins);
                if (b >= nbins) b = nbins - 1;
                local[b]++;
            }

            /* Merge the private histogram into the shared one.  Each bin is
             * updated with an atomic add so no critical section serialises the
             * whole merge. */
            for (int b = 0; b < nbins; b++) {
                if (local[b] != 0) {
#pragma omp atomic
                    counts[b] += local[b];
                }
            }

            free(local);
        }
    } /* end parallel */
}
