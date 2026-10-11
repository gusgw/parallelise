#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "kernel_b.h"
#include "problem.h"

void kernel_b(double *u, long n, int iters)
{
    double *un = malloc(sizeof(double) * n * n);
    if (!un) return;

    for (int iter = 0; iter < iters; iter++) {
        memcpy(un, u, sizeof(double) * n * n);

        #pragma omp parallel for schedule(static)
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
