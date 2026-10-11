#include "kernel_b.h"
#include <omp.h>
#include <stdlib.h>

void kernel_b(double *u, long n, int iters) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        u[i] = i; // initialize u
    }
    
    for (int iter = 0; iter < iters; iter++) {
        double *v = (double *) malloc(n * sizeof(double));
        #pragma omp parallel for
        for (int i = 1; i < n - 1; i++) {
            v[i] = (u[i - 1] + u[i + 1]) / 2; // update u
        }
        v[0] = u[0];
        v[n-1] = u[n-1];
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            u[i] = v[i];
        }
        free(v);
    }
}
