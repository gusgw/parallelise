/*
 * test_cg.c  --  Unit test for the GPU conjugate-gradient kernel.
 *
 * Checks that cgsolve() reproduces the Python-notebook oracle checksum
 * (REF_CHECKSUM) within REF_TOL, using the same deterministic RHS (b_of).
 */

#include <stdio.h>
#include <stdlib.h>

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

    /* fill RHS with the deterministic pattern from problem.h */
    for (long t = 0; t < n; t++)
        b[t] = b_of(t);

    /* run the GPU solver */
    cgsolve(x, b, m, iters);

    /* compute checksum and compare to the notebook oracle */
    double cs = checksum(x, m);
    printf("checksum = %.10g  (want %.10g  tol %g)\n",
           cs, (double)REF_CHECKSUM, (double)REF_TOL);

    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "CG checksum matches oracle");

    free(b);
    free(x);
    return TEST_SUMMARY();
}
