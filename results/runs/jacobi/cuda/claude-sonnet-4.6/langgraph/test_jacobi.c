/*
 * test_jacobi.c  –  unit test for the GPU Jacobi kernel.
 *
 * Fills an n×n grid with the canonical initial condition (problem.h),
 * runs jacobi() for REF_ITERS sweeps, computes the checksum, and checks
 * it against REF_CHECKSUM within REF_TOL (all from reference.h).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "jacobi.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(void)
{
    long n     = REF_N;
    int  iters = (int)REF_ITERS;

    double *u = (double *)malloc(sizeof(double) * (size_t)n * (size_t)n);
    if (!u) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* Fill initial / boundary condition */
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            u[i * n + j] = u_init(i, j, n);

    /* Run the GPU kernel */
    jacobi(u, n, iters);

    /* Compute checksum and compare to oracle */
    double cs = checksum(u, n);

    printf("checksum = %.13g  (ref = %.13g  tol = %g)\n",
           cs, (double)REF_CHECKSUM, (double)REF_TOL);

    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                "jacobi checksum matches oracle");

    free(u);
    return TEST_SUMMARY();
}
