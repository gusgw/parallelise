/* jacobi.c — 2-D Jacobi heat stencil, parallelised with OpenMP.
 *
 * Algorithm (from jacobi.ipynb):
 *   - n×n row-major grid, top row (i==0) fixed at 1.0, other edges fixed at 0.0.
 *   - Each of `iters` sweeps replaces every INTERIOR cell by the average of its
 *     four axis-aligned neighbours, using the PREVIOUS sweep's values (double-buffer).
 *   - Sweeps are sequential (each depends on the previous); the inner loop over
 *     interior rows is parallelised with OpenMP.
 */

#include <stdlib.h>   /* malloc, free  */
#include <string.h>   /* memcpy        */

#include "jacobi.h"

void jacobi(double *u, long n, int iters)
{
    /* Allocate a scratch buffer for the double-buffer swap. */
    double *tmp = (double *)malloc(sizeof(double) * n * n);
    if (!tmp) return;   /* nothing sensible to do without memory */

    /* Copy the initial grid into tmp so the first swap is consistent. */
    memcpy(tmp, u, sizeof(double) * n * n);

    /* Pointers that we swap each sweep instead of copying the whole grid. */
    double *cur  = u;
    double *next = tmp;

    for (int iter = 0; iter < iters; iter++) {

        /* Copy boundary rows/columns from cur into next so the fixed edges
         * are always present in the destination buffer without extra branches
         * inside the hot loop.  Only four thin strips need copying:
         *   row 0, row n-1, column 0, column n-1.
         * We copy the full top and bottom rows, then fix the left/right edges
         * of every interior row. */
        memcpy(next,               cur,               sizeof(double) * n); /* row 0   */
        memcpy(next + (n-1)*n,     cur + (n-1)*n,     sizeof(double) * n); /* row n-1 */

        /* Interior rows: update stencil in parallel, then fix left/right edges. */
#pragma omp parallel for schedule(static)
        for (long i = 1; i < n - 1; i++) {
            /* Fix the two boundary columns for this row. */
            next[i * n + 0    ] = cur[i * n + 0    ];
            next[i * n + (n-1)] = cur[i * n + (n-1)];

            /* Stencil over interior columns. */
            for (long j = 1; j < n - 1; j++) {
                next[i * n + j] = 0.25 * (cur[(i-1)*n + j] +
                                           cur[(i+1)*n + j] +
                                           cur[i*n + (j-1)] +
                                           cur[i*n + (j+1)]);
            }
        }

        /* Swap buffers: next becomes the new current state. */
        double *swap = cur;
        cur  = next;
        next = swap;
    }

    /* After all sweeps, `cur` holds the final grid.  If it ended up in the
     * scratch buffer (tmp) we need to copy it back into u. */
    if (cur != u) {
        memcpy(u, cur, sizeof(double) * n * n);
    }

    free(tmp);
}
