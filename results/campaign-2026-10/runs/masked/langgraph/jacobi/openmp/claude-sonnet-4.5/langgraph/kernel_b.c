#include "kernel_b.h"
#include <stdlib.h>
#include <string.h>

void kernel_b(double *u, long n, int iters)
{
    double *un = (double *)malloc(n * n * sizeof(double));
    
    for (int iter = 0; iter < iters; iter++) {
        // Copy u to un
        memcpy(un, u, n * n * sizeof(double));
        
        // Update interior points
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
