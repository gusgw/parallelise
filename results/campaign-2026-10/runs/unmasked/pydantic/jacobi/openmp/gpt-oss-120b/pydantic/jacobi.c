#include <stdlib.h>
#include <string.h>
#include "jacobi.h"

/* Parallel OpenMP implementation of the 2D Jacobi heat stencil.
 * The grid is stored in row‑major order, size n × n.
 * The four edges are fixed boundaries and must never be overwritten.
 * For each sweep we compute a new grid (new_u) from the previous grid (u)
 * using the 5‑point stencil and then copy the result back into u.
 */

void jacobi(double *u, long n, int iters)
{
    /* Allocate a temporary buffer for the new values. */
    double *new_u = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    if (!new_u) return; /* allocation failure – nothing we can do */

    const size_t size = (size_t)n * (size_t)n * sizeof(double);

    for (int it = 0; it < iters; ++it) {
        /* Preserve the fixed boundary values by copying the whole array. */
        memcpy(new_u, u, size);

        /* Update interior points. */
        #pragma omp parallel for schedule(static) default(none) shared(u,new_u,n)
        for (long i = 1; i < n - 1; ++i) {
            /* Compute the offset of the first element of row i. */
            size_t row_offset = (size_t)i * (size_t)n;
            for (long j = 1; j < n - 1; ++j) {
                new_u[row_offset + j] = 0.25 * (
                    u[row_offset - n + j] +   /* u[i-1][j] */
                    u[row_offset + n + j] +   /* u[i+1][j] */
                    u[row_offset + j - 1] +   /* u[i][j-1] */
                    u[row_offset + j + 1]     /* u[i][j+1] */
                );
            }
        }

        /* Copy the updated interior back into the original array for the next sweep. */
        memcpy(u, new_u, size);
    }

    free(new_u);
}
