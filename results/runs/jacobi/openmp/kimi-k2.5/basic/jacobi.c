#include "jacobi.h"
#include <stdlib.h>
#include <string.h>

void jacobi(double *u, long n, int iters)
{
    /* Allocate a temporary buffer for double-buffering */
    double *un = (double *)malloc(n * n * sizeof(double));
    if (!un) return;

    /* Copy initial state to un (keeps fixed boundaries) */
    memcpy(un, u, n * n * sizeof(double));

    for (int iter = 0; iter < iters; iter++) {
        /* Update interior cells: un[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]) */
        #pragma omp parallel for schedule(static)
        for (long i = 1; i < n - 1; i++) {
            for (long j = 1; j < n - 1; j++) {
                un[i * n + j] = 0.25 * (u[(i - 1) * n + j] +
                                        u[(i + 1) * n + j] +
                                        u[i * n + (j - 1)] +
                                        u[i * n + (j + 1)]);
            }
        }

        /* Swap pointers for next iteration */
        double *tmp = u;
        u = un;
        un = tmp;
    }

    /* If we did an odd number of iterations, the final result is in un, not u.
     * Copy it back to the original buffer. */
    if (iters % 2 == 1) {
        memcpy(u, un, n * n * sizeof(double));
    }

    free(un);
}
