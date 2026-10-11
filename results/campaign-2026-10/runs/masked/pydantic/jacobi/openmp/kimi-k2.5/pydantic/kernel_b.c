#include "kernel_b.h"
#include "problem.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void kernel_b(double *u, long n, int iters)
{
    double *un = malloc(n * n * sizeof(double));
    if (!un) return;
    
    /* Initialize u using the problem's u_init function */
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }
    
    /* Jacobi iterations */
    for (int iter = 0; iter < iters; iter++) {
        /* Copy current state to un */
        memcpy(un, u, n * n * sizeof(double));
        
        /* Update interior points */
        #pragma omp parallel for
        for (long i = 1; i < n - 1; i++) {
            for (long j = 1; j < n - 1; j++) {
                u[i * n + j] = 0.25 * (un[(i - 1) * n + j] + 
                                       un[(i + 1) * n + j] + 
                                       un[i * n + (j - 1)] + 
                                       un[i * n + (j + 1)]);
            }
        }
    }
    
    free(un);
}
