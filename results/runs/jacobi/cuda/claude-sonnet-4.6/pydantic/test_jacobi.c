/*
 * test_jacobi.c  –  unit test for the GPU Jacobi kernel.
 *
 * Fills an n x n grid with the canonical initial condition (top row = 1.0,
 * everything else = 0.0), runs `iters` sweeps via the GPU kernel, computes
 * the checksum, and checks it against the notebook oracle within REF_TOL.
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
    long  n     = REF_N;
    int   iters = (int)REF_ITERS;
    long  total = n * n;

    /* Allocate and fill the initial grid */
    double *u = (double *)malloc(sizeof(double) * (size_t)total);
    if (!u) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

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

    /* Sanity: boundary rows must remain unchanged */
    int boundary_ok = 1;
    for (long j = 0; j < n; j++) {
        if (u[0 * n + j] != 1.0)          { boundary_ok = 0; break; }  /* top hot */
        if (u[(n-1) * n + j] != 0.0)      { boundary_ok = 0; break; }  /* bottom cold */
        if (u[j * n + 0] != u_init(j,0,n)){ boundary_ok = 0; break; }  /* left */
        if (u[j * n + (n-1)] != u_init(j,n-1,n)){ boundary_ok = 0; break; } /* right */
    }
    CHECK(boundary_ok, "boundary cells are unchanged after sweeps");

    free(u);
    return TEST_SUMMARY();
}
