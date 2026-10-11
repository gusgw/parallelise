/* jacobi.c — 2-D Jacobi heat stencil, parallelised with OpenMP.
 *
 * Algorithm (from jacobi.ipynb):
 *   - n x n row-major grid; top row (i==0) is hot (1.0), all other edges cold (0.0).
 *   - For each of `iters` sweeps every INTERIOR cell is replaced by the average of its
 *     four neighbours from the PREVIOUS sweep (double-buffered):
 *         u_new[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1])
 *   - The four boundary rows/columns are fixed and never modified.
 *
 * Parallelism: the inner loop over interior rows is independent across rows within a
 * single sweep, so we parallelise it with '#pragma omp parallel for'.  Sweeps are
 * sequential because each depends on the previous one.
 */

#include <stdlib.h>   /* malloc, free  */
#include <string.h>   /* memcpy        */

#include "jacobi.h"

void jacobi(double *u, long n, int iters)
{
    /* Allocate a scratch buffer for the double-buffered "new" values. */
    double *scratch = (double *)malloc(sizeof(double) * n * n);
    if (!scratch) return;   /* allocation failure */

    /* We use two pointers that alternate roles each sweep:
     *   cur  — the grid from the previous sweep (read-only inside the hot loop)
     *   nxt  — the grid being written this sweep
     * We start with cur == u (the caller's buffer) so that after an even number of
     * sweeps the result ends up back in u automatically.  After an odd number of sweeps
     * the result is in scratch, so we copy it back to u before returning. */
    double *cur = u;
    double *nxt = scratch;

    for (int iter = 0; iter < iters; iter++) {

        /* Copy the whole grid into nxt so that boundary cells are already correct
         * and we never have to special-case them inside the hot loop. */
        memcpy(nxt, cur, sizeof(double) * n * n);

        /* Update every interior cell using the four-neighbour average.
         * Rows are independent of each other → parallelise over i. */
#pragma omp parallel for schedule(static)
        for (long i = 1; i < n - 1; i++) {
            for (long j = 1; j < n - 1; j++) {
                nxt[i * n + j] = 0.25 * (cur[(i - 1) * n + j] +
                                          cur[(i + 1) * n + j] +
                                          cur[i * n + (j - 1)] +
                                          cur[i * n + (j + 1)]);
            }
        }

        /* Swap roles: the freshly written buffer becomes the source for the next sweep. */
        double *tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    /* After `iters` swaps:
     *   iters even → cur == u       (result already in caller's buffer, scratch is nxt)
     *   iters odd  → cur == scratch (result in scratch, caller's buffer is nxt)
     * In the odd case copy the result back into the caller's buffer. */
    if (iters % 2 != 0) {
        /* cur == scratch holds the result; nxt == u is the caller's buffer. */
        memcpy(u, cur, sizeof(double) * n * n);
    }

    free(scratch);
}
