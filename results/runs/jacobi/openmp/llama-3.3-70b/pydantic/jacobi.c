#include "jacobi.h"
#include "reference.h"
#include <omp.h>
#include <stdlib.h>

void jacobi(double *u, long n, int iters) {
    double *un = malloc(n * n * sizeof(double));
    for (int sweep = 0; sweep < iters; sweep++) {
        #pragma omp parallel for
        for (long i = 1; i < n - 1; i++) {
            for (long j = 1; j < n - 1; j++) {
                un[i * n + j] = 0.25 * (u[(i-1) * n + j] + u[(i+1) * n + j] + u[i * n + j - 1] + u[i * n + j + 1]);
            }
        }
        double *tmp = u;
        u = un;
        un = tmp;
    }
    free(un);
}
