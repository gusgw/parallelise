#include "kernel_b.h"
#include <stdlib.h>
#include <string.h>

void kernel_b(double *u, long n, int iters)
{
    double *un = malloc(sizeof(double) * n * n);
    if (!un) return;

    for (int iter = 0; iter < iters; iter++) {
        memcpy(un, u, sizeof(double) * n * n);
        
        #pragma omp parallel for
        for (long i = 1; i < n - 1; i++) {
            for (long j = 1; j < n - 1; j++) {
                un[i * n + j] = 0.25 * (u[(i - 1) * n + j] + 
                                        u[(i + 1) * n + j] + 
                                        u[i * n + (j - 1)] + 
                                        u[i * n + (j + 1)]);
            }
        }
        
        memcpy(u, un, sizeof(double) * n * n);
    }

    free(un);
}
