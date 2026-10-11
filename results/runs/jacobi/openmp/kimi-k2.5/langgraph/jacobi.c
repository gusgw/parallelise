#include "jacobi.h"
#include <stdlib.h>
#include <string.h>

void jacobi(double *u, long n, int iters)
{
    /* Allocate temporary buffer for double-buffering */
    double *u_new = (double *)malloc(n * n * sizeof(double));
    if (!u_new) return;
    
    /* Copy initial state to u_new (keeps boundaries) */
    memcpy(u_new, u, n * n * sizeof(double));
    
    for (int iter = 0; iter < iters; iter++) {
        /* Parallelize the interior row updates */
        #pragma omp parallel for
        for (long i = 1; i < n - 1; i++) {
            for (long j = 1; j < n - 1; j++) {
                u_new[i * n + j] = 0.25 * (
                    u[(i - 1) * n + j] +
                    u[(i + 1) * n + j] +
                    u[i * n + (j - 1)] +
                    u[i * n + (j + 1)]
                );
            }
        }
        
        /* Swap pointers for next iteration */
        double *tmp = u;
        u = u_new;
        u_new = tmp;
    }
    
    /* If iters is odd, final result is in u_new, so copy back to original u */
    if (iters % 2 != 0) {
        memcpy(u_new, u, n * n * sizeof(double));
    }
    
    free(u_new);
}
