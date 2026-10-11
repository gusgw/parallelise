#include <stdio.h>
#include <stdlib.h>
#include "jacobi.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;
    int iters = (int)REF_ITERS;

    /* Allocate and initialize the grid */
    double *u = (double *)malloc(n * n * sizeof(double));
    if (!u) {
        printf("FAIL: could not allocate grid\n");
        return 1;
    }

    /* Initialize with boundary conditions: top row hot (1.0), rest cold (0.0) */
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    /* Run the Jacobi iteration */
    jacobi(u, n, iters);

    /* Compute checksum and verify against reference */
    double cs = checksum(u, n);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum mismatch");

    free(u);

    return TEST_SUMMARY();
}
