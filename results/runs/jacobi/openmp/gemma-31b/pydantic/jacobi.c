#include "jacobi.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void jacobi(double *u, long n, int iters) {
    if (n < 3) return;
    double *u_new = (double *)malloc(n * n * sizeof(double));
    if (!u_new) return;

    memcpy(u_new, u, n * n * sizeof(double));

    for (int iter = 0; iter < iters; iter++) {
        #pragma omp parallel for
        for (long i = 1; i < n - 1; i++) {
            for (long j = 1; j < n - 1; j++) {
                u_new[i * n + j] = 0.25 * (u[(i - 1) * n + j] + 
                                          u[(i + 1) * n + j] + 
                                          u[i * n + (j - 1)] + 
                                          u[i * n + (j + 1)]);
            }
        }
        memcpy(u, u_new, n * n * sizeof(double));
    }

    free(u_new);
}
