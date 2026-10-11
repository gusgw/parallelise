/*
 * test_cg.c  –  unit test for the GPU CG solver.
 *
 * Builds with nvcc (plain C, no MPI, no OpenMP).
 * Checks that the checksum of the solution after REF_ITERS iterations on the
 * REF_M x REF_M grid matches the Python-notebook oracle within REF_TOL.
 */

#include <stdio.h>
#include <stdlib.h>

#include "cg.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    const long m     = REF_M;
    const long n     = m * m;
    const int  iters = (int)REF_ITERS;

    /* allocate and fill the right-hand side */
    double *b = (double *)malloc(sizeof(double) * n);
    double *x = (double *)malloc(sizeof(double) * n);
    if (!b || !x) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    for (long idx = 0; idx < n; idx++)
        b[idx] = b_of(idx);

    /* run the GPU solver */
    cgsolve(x, b, m, iters);

    /* compute checksum and compare to the oracle */
    double cs = checksum(x, m);
    printf("checksum = %.10g  (want %.10g, tol %g)\n",
           cs, (double)REF_CHECKSUM, (double)REF_TOL);

    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL,
                "GPU CG checksum matches notebook oracle");

    free(b);
    free(x);
    return TEST_SUMMARY();
}
