#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jacobi.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;
    int iters = (int)REF_ITERS;
    double *u = malloc(sizeof(double) * n * n);
    if (!u) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* Initialize grid: top row hot (1.0), rest cold (0.0) */
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }

    /* Run Jacobi iterations on GPU */
    jacobi(u, n, iters);

    /* Compute checksum */
    double cs = checksum(u, n);

    /* Check against reference */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum mismatch");

    free(u);
    return TEST_SUMMARY();
}
