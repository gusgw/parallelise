#include <stdlib.h>
#include <stdio.h>
#include "kernel_b.h"
#include "reference.h"
#include "testing.h"
#include "problem.h"

int main(void)
{
    long n = REF_N;
    int iters = (int)REF_ITERS;
    size_t sz = (size_t)n * (size_t)n;
    double *u = (double *)malloc(sz * sizeof(double));
    if (!u) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    // initialise
    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < n; ++j) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    kernel_b(u, n, iters);

    double chk = checksum(u, n);
    CHECK_CLOSE(chk, REF_CHECKSUM, REF_TOL, "checksum matches reference");

    free(u);
    return TEST_SUMMARY();
}
