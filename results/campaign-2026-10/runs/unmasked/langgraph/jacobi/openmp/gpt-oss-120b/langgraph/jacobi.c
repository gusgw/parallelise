#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "jacobi.h"

/* Implements the 2D Jacobi heat stencil using double buffering.
 * The grid is stored in row‑major order, size n×n.
 * The four edges are fixed boundaries and never modified.
 */
void jacobi(double *u, long n, int iters)
{
    if (n <= 2 || iters <= 0) return; // nothing to do for degenerate cases

    size_t sz = (size_t)n * (size_t)n * sizeof(double);
    double *buf = (double *)malloc(sz);
    if (!buf) {
        /* allocation failure – abort as we cannot continue */
        abort();
    }

    /* Initialise the auxiliary buffer with the same boundary values as u.
     * The interior will be overwritten during the sweeps.
     */
    memcpy(buf, u, sz);

    double *src = u;   /* current grid */
    double *dst = buf; /* next grid */

    for (int it = 0; it < iters; ++it) {
        /* Compute interior points only; boundaries stay unchanged. */
        #pragma omp parallel for schedule(static)
        for (long i = 1; i < n - 1; ++i) {
            long base = i * n;
            for (long j = 1; j < n - 1; ++j) {
                dst[base + j] = 0.25 * (src[(i - 1) * n + j] +
                                         src[(i + 1) * n + j] +
                                         src[base + j - 1] +
                                         src[base + j + 1]);
            }
        }
        /* Swap source and destination pointers for the next iteration. */
        double *tmp = src;
        src = dst;
        dst = tmp;
    }

    /* After the loop, 'src' points to the most recent grid.
     * If it is not the original array 'u', copy the result back.
     */
    if (src != u) {
        memcpy(u, src, sz);
    }

    free(buf);
}
