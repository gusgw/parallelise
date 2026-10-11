/*
 * test_cg.c  –  unit test for cgsolve (cg.cu / cg.h).
 *
 * Builds as plain C (compiled by nvcc but no CUDA calls here).
 * Checks the checksum of the solution against the notebook oracle (reference.h)
 * within REF_TOL.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "cg.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long m     = REF_M;
    int  iters = (int)REF_ITERS;
    long n     = m * m;

    double *b = (double *)malloc(sizeof(double) * n);
    double *x = (double *)malloc(sizeof(double) * n);
    if (!b || !x) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    /* deterministic right-hand side (same as the Python notebook) */
    for (long idx = 0; idx < n; idx++)
        b[idx] = b_of(idx);

    /* run the GPU solver */
    cgsolve(x, b, m, iters);

    /* compute checksum and compare to the notebook oracle */
    double cs = checksum(x, m);

    printf("checksum = %.10g  (expected %.10g, tol %g)\n",
           cs, (double)REF_CHECKSUM, (double)REF_TOL);

    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                "cgsolve checksum matches notebook oracle");

    free(b);
    free(x);
    return TEST_SUMMARY();
}
