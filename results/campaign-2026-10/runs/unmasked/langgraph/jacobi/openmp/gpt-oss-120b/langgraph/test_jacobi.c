#include <stdio.h>
#include <stdlib.h>
#include "jacobi.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;
    size_t sz = (size_t)n * (size_t)n * sizeof(double);
    double *u = (double *)malloc(sz);
    if (!u) {
        perror("malloc");
        return 1;
    }

    /* initialise grid with hot top row and cold elsewhere */
    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < n; ++j) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    jacobi(u, n, (int)REF_ITERS);

    double cs = checksum(u, n);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "Jacobi checksum");

    free(u);
    return TEST_SUMMARY();
}
