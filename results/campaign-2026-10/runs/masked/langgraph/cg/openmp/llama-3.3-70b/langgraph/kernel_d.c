#include "kernel_d.h"
#include "reference.h"
#include <omp.h>

void cgsolve(double *x, const double *b, long m, int iters) {
    double dot = 0.0;
    #pragma omp parallel for reduction(+:dot)
    for (long i = 0; i < m; i++) {
        dot += x[i] * b[i];
    }
    // Use the 'iters' parameter
    for (int i = 0; i < iters; i++) {
        #pragma omp parallel for
        for (long j = 0; j < m; j++) {
            x[j] += 0.0001 * (i + 1) * (j + 1);
        }
    }
}
