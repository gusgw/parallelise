#include <stdlib.h>
#include <string.h>
#include "jacobi.h"

void jacobi(double *u, long n, int iters)
{
    double *u_new = malloc(sizeof(double) * n * n);
    if (!u_new) return;
    
    for (int iter = 0; iter < iters; iter++) {
        /* Copy current grid to u_new (keeps fixed boundaries) */
        memcpy(u_new, u, sizeof(double) * n * n);
        
        /* Update interior cells using OpenMP parallel for */
        #pragma omp parallel for schedule(static)
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
    
    /* If final result is in u_new, copy back to original u */
    if (u != u_new) {
        memcpy(u_new, u, sizeof(double) * n * n);
    }
    
    free(u_new);
}
