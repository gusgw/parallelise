#include "jacobi.h"
#include <stdlib.h>
#include <string.h>

void jacobi(double *u, long n, int iters)
{
    double *u_new = malloc(n * n * sizeof(double));
    if (!u_new) return;
    
    memcpy(u_new, u, n * n * sizeof(double));
    
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
    
    if (iters % 2 == 1) {
        memcpy(u_new, u, n * n * sizeof(double));
    }
    
    free(u_new);
}
