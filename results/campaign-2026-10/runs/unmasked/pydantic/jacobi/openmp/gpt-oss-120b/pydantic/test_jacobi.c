#include <stdlib.h>
#include <stdio.h>
#include "jacobi.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;
    size_t total = (size_t)n * (size_t)n;
    double *u = (double *)malloc(total * sizeof(double));
    if (!u) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* Initialize grid with hot top row and cold elsewhere. */
    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < n; ++j) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    jacobi(u, n, (int)REF_ITERS);

    double cs = checksum(u, n);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum matches reference");

    free(u);
    return TEST_SUMMARY();
}
