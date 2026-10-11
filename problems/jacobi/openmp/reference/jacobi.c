#include <stdlib.h>
#include <string.h>

#include "jacobi.h"

/* Reference solution (validates the harness -- NOT shown to the agents). Double-buffer
 * the grid; each sweep parallelises over the interior rows (independent given the
 * previous grid); pointer-swap between sweeps and copy back if the final grid landed in
 * the scratch buffer. */
void jacobi(double *u, long n, int iters)
{
    double *scratch = malloc(sizeof(double) * n * n);
    memcpy(scratch, u, sizeof(double) * (size_t)n * n);  /* fixed boundaries */

    double *cur = u, *next = scratch;
    for (int it = 0; it < iters; it++) {
        #pragma omp parallel for
        for (long i = 1; i < n - 1; i++)
            for (long j = 1; j < n - 1; j++)
                next[i * n + j] = 0.25 * (cur[(i - 1) * n + j] + cur[(i + 1) * n + j]
                                          + cur[i * n + j - 1] + cur[i * n + j + 1]);
        double *tmp = cur; cur = next; next = tmp;
    }
    if (cur != u)
        memcpy(u, cur, sizeof(double) * (size_t)n * n);
    free(scratch);
}
