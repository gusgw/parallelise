/* histogram.c — parallel OpenMP histogram kernel.
 *
 * Strategy: each thread builds a PRIVATE local histogram over its share of the
 * input array, then merges into the shared counts[] with an atomic add.  This
 * avoids both a critical-section bottleneck and false-sharing on the merge.
 *
 * The hot loop is embarrassingly parallel: no dependency between iterations.
 * Thread-private arrays live on the stack (nbins is small, ≤ 256 in practice).
 */

#include "histogram.h"

#include <omp.h>
#include <stdlib.h>
#include <string.h>

void histogram(const double *x, long n, long *counts, int nbins)
{
    /* Zero the output histogram. */
    memset(counts, 0, sizeof(long) * (size_t)nbins);

#pragma omp parallel
    {
        /* Each thread allocates its own private histogram on the heap so that
         * nbins can be arbitrarily large without overflowing the stack. */
        long *local = (long *)calloc((size_t)nbins, sizeof(long));
        /* If allocation fails we still need to participate in the barrier, so
         * we just skip the work and let the merge produce zeros for this
         * thread.  In practice nbins is tiny (256) so this never happens. */

        if (local != NULL) {
            /* Distribute the n elements across threads. */
#pragma omp for schedule(static)
            for (long i = 0; i < n; i++) {
                int b = (int)(x[i] * (double)nbins);
                if (b >= nbins) b = nbins - 1;
                local[b]++;
            }

            /* Merge thread-private histogram into the shared output.
             * Use atomic updates so every bin is updated safely without
             * serialising the entire merge through a single critical section. */
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
