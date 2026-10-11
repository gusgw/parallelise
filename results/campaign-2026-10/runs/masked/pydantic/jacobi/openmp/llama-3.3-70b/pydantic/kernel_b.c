#include "kernel_b.h"
#include "problem.h"
#include <omp.h>
#include <stdlib.h>

void kernel_b(double *u, long n, int iters) {
    double *un = (double *)malloc(n * n * sizeof(double));
    #pragma omp parallel for
    for (long i = 0; i < n * n; i++) {
        u[i] = u_init(i / n, i % n, n);
    }
    for (int t = 0; t < iters; t++) {
        #pragma omp parallel for
        for (long i = 0; i < n * n; i++) {
            un[i] = u[i];
        }
        #pragma omp parallel for
        for (long i = 0; i < n * n; i++) {
            u[i] = un[i] * 2.0;
        }
    }
    free(un);
}
