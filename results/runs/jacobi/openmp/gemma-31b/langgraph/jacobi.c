#include "jacobi.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void jacobi(double *u, long n, int iters) {
    if (n < 3) return;
    double *u_new = (double *)malloc(n * n * sizeof(double));
    if (!u_new) return;

    memcpy(u_new, u, n * n * sizeof(double));

    double *src = u;
    double *dst = u_new;

    for (int iter = 0; iter < iters; iter++) {
        #pragma omp parallel for
        for (long i = 1; i < n - 1; i++) {
            for (long j = 1; j < n - 1; j++) {
                dst[i * n + j] = 0.25 * (src[(i - 1) * n + j] + 
                                        src[(i + 1) * n + j] + 
                                        src[i * n + (j - 1)] + 
                                        src[i * n + (j + 1)]);
            }
        }
        // Swap buffers
        double *tmp = src;
        src = dst;
        dst = tmp;
    }

    // Ensure final result is in u
    if (src != u) {
        memcpy(u, src, n * n * sizeof(double));
    }

    free(u_new);
}
