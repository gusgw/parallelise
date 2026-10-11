#include "jacobi.h"
#include <omp.h>
#include <stdlib.h>

void jacobi(double *u, long n, int iters) {
    double *un = (double *)malloc(n * n * sizeof(double));
    #pragma omp parallel for
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            un[i * n + j] = u[i * n + j];
        }
    }
    for (int sweep = 0; sweep < iters; sweep++) {
        #pragma omp parallel for
        for (long i = 1; i < n - 1; i++) {
            for (long j = 1; j < n - 1; j++) {
                un[i * n + j] = 0.25 * (u[(i-1) * n + j] + u[(i+1) * n + j] + u[i * n + j - 1] + u[i * n + j + 1]);
            }
        }
        #pragma omp parallel for
        for (long i = 0; i < n; i++) {
            for (long j = 0; j < n; j++) {
                u[i * n + j] = un[i * n + j];
            }
        }
    }
    free(un);
}
