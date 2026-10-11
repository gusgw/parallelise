#include "jacobi.h"
#include <stdlib.h>
#include <string.h>

void jacobi(double *u, long n, int iters)
{
    /* Allocate temporary buffer for double-buffering */
    double *un = (double *)malloc(n * n * sizeof(double));
    if (!un) return;
    
    /* Copy initial state to un (keeps fixed boundaries) */
    memcpy(un, u, n * n * sizeof(double));
    
    for (int iter = 0; iter < iters; iter++) {
        /* Parallelize the interior row updates with OpenMP */
        #pragma omp parallel for
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
    
    /* If iters is odd, the final result is in un, so copy back to u */
    if (iters % 2 != 0) {
        memcpy(u, un, n * n * sizeof(double));
    }
    
    free(un);
}
