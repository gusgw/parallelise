/* jacobi.c — parallel 2-D Jacobi heat stencil (OpenMP).
 *
 * Algorithm (from jacobi.ipynb):
 *   - n x n row-major grid; top row (i==0) is hot (1.0), all other edges cold (0.0).
 *   - For each of `iters` sweeps, every INTERIOR cell is replaced by the average of
 *     its four neighbours from the PREVIOUS sweep (double-buffered):
 *         u_new[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])
 *   - The four boundary rows/columns are fixed and never modified.
 *   - Sweeps are sequential (each depends on the previous); the inner loop over
 *     interior rows is parallelised with OpenMP.
 */

#include <stdlib.h>
#include <string.h>

#include "jacobi.h"

void jacobi(double *u, long n, int iters)
{
    /* Allocate a scratch buffer the same size as the caller's grid. */
    double *buf = (double *)malloc(sizeof(double) * n * n);
    if (!buf) return;

    /* src always points to the grid we READ from; dst to the grid we WRITE to.
     * We start by reading from the caller's array u. */
    double *src = u;
    double *dst = buf;

    for (int iter = 0; iter < iters; iter++) {
        /* Copy src -> dst so that boundary rows/columns are preserved in dst. */
        memcpy(dst, src, sizeof(double) * n * n);

        /* Update every interior cell using the four neighbours from src. */
#pragma omp parallel for schedule(static)
        for (long i = 1; i < n - 1; i++) {
            for (long j = 1; j < n - 1; j++) {
                dst[i * n + j] = 0.25 * (src[(i - 1) * n + j] +
                                          src[(i + 1) * n + j] +
                                          src[i * n + (j - 1)] +
                                          src[i * n + (j + 1)]);
            }
        }

        /* Swap src/dst for the next sweep. */
        double *tmp = src;
        src = dst;
        dst = tmp;
    }

    /* After `iters` swaps, src holds the final result.
     * If iters is odd  -> src == buf  (need to copy back into u)
     * If iters is even -> src == u    (already in the right place) */
    if (src != u) {
        memcpy(u, src, sizeof(double) * n * n);
    }

    free(buf);
}
