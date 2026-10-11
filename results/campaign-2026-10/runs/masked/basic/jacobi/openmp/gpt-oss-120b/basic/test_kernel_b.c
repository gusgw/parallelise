#include <stdlib.h>
#include <stdio.h>

#include "kernel_b.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(void)
{
    const long n = REF_N;
    const int iters = REF_ITERS;
    const size_t size = (size_t)n * (size_t)n;

    double *u = (double *)malloc(size * sizeof(double));
    if (!u) {
        perror("malloc");
        return 1;
    }

    /* Initialise the grid using the helper from problem.h */
    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < n; ++j) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    kernel_b(u, n, iters);

    double cs = checksum(u, n);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");

    free(u);
    return TEST_SUMMARY();
}
