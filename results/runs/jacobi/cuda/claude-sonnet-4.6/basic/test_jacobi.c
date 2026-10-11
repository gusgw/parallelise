/*
 * test_jacobi.c  --  Unit test for the GPU Jacobi kernel.
 *
 * Fills an n x n grid with the standard initial condition (top row = 1.0,
 * rest = 0.0), runs `iters` Jacobi sweeps via the GPU kernel, computes the
 * checksum, and checks it against the notebook oracle within REF_TOL.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "jacobi.h"
#include "problem.h"
#include "reference.h"
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

    /* Fill with the standard initial / boundary condition */
    for (long i = 0; i < n; i++)
        for (long j = 0; j < n; j++)
            u[i * n + j] = u_init(i, j, n);

    /* Run the GPU kernel */
    jacobi(u, n, iters);

    /* Compute checksum and compare to oracle */
    double cs = checksum(u, n);

    printf("checksum = %.13g\n", cs);
    printf("expected = %.13g\n", (double)REF_CHECKSUM);
    printf("diff     = %.6g  (tol = %g)\n",
           fabs(cs - REF_CHECKSUM), (double)REF_TOL);

    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                "GPU Jacobi checksum matches notebook oracle");

    free(u);
    return TEST_SUMMARY();
}
