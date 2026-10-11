#include "jacobi.h"
#include <stdlib.h>
#include <string.h>

/* Jacobi 5-point heat stencil on an n x n row-major grid. For `iters` sweeps, replace
 * each INTERIOR cell by the average of its four neighbours, using the PREVIOUS sweep's
 * values (double-buffered):
 *     u_new[i][j] = 0.25 * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1]).
 * The four edges are FIXED boundaries and are never modified. On return u holds the
 * final grid.
 */
void jacobi(double *u, long n, int iters)
{
    /* Allocate temporary buffer for double-buffering */
    double *un = (double *)malloc(n * n * sizeof(double));
    
    for (int iter = 0; iter < iters; iter++) {
        /* Copy u to un (keeps the fixed boundaries) */
        memcpy(un, u, n * n * sizeof(double));
        
        /* Update interior cells in parallel */
        #pragma omp parallel for
        for (long i = 1; i < n - 1; i++) {
            for (long j = 1; j < n - 1; j++) {
                un[i * n + j] = 0.25 * (u[(i-1) * n + j] + u[(i+1) * n + j] + 
                                        u[i * n + (j-1)] + u[i * n + (j+1)]);
            }
        }
        
        /* Copy result back to u */
        memcpy(u, un, n * n * sizeof(double));
    }
    
    free(un);
}
