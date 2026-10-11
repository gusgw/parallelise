#include "kernel_b.h"
#include <omp.h>
#include "problem.h"

void kernel_b(double *u, long n, int iters) {
    #pragma omp parallel for
    for (long i = 0; i < n * n; i++) {
        u[i] = u_init(i / n, i % n, n) * (1 + iters);
    }
}
