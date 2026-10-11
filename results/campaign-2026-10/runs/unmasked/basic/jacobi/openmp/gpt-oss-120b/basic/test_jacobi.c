#include <stdio.h>
#include <stdlib.h>
#include "jacobi.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(void) {
    long n = REF_N;
    int iters = (int)REF_ITERS;
    size_t size = (size_t)n * (size_t)n;
    double *u = (double *)malloc(size * sizeof(double));
    if (!u) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }
    // initialise grid using u_init
    for (long i = 0; i < n; ++i) {
        for (long j = 0; j < n; ++j) {
            u[i * n + j] = u_init(i, j, n);
        }
    }
    jacobi(u, n, iters);
    double chk = checksum(u, n);
    CHECK_CLOSE(chk, REF_CHECKSUM, REF_TOL, "checksum");
    free(u);
    return TEST_SUMMARY();
}
