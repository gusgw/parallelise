#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "jacobi.h"

void jacobi(double *u, long n, int iters)
{
    double *u_new = malloc(sizeof(double) * n * n);
    if (!u_new) return;
    
    memcpy(u_new, u, sizeof(double) * n * n);
    
    for (int iter = 0; iter < iters; iter++) {
        #pragma omp parallel for schedule(static)
        for (long i = 1; i < n - 1; i++) {
            for (long j = 1; j < n - 1; j++) {
                u_new[i * n + j] = 0.25 * (u[(i - 1) * n + j] +
                                            u[(i + 1) * n + j] +
                                            u[i * n + (j - 1)] +
                                            u[i * n + (j + 1)]);
            }
        }
        
        double *tmp = u;
        u = u_new;
        u_new = tmp;
    }
    
    if (iters % 2 != 0) {
        memcpy(u, u_new, sizeof(double) * n * n);
    }
    
    free(u_new);
}
