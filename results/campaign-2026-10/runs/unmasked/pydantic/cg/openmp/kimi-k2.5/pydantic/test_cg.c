#include "cg.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"
#include <stdlib.h>
#include <stdio.h>

int main(void)
{
    long m = REF_M;
    int iters = (int)REF_ITERS;
    long n = m * m;

    /* Allocate and fill b using the deterministic b_of function */
    double *b = (double *)malloc(n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));

    for (long k = 0; k < n; k++) {
        b[k] = b_of(k);
    }

    /* Run the conjugate gradient solver */
    cgsolve(x, b, m, iters);

    /* Compute checksum */
    double cs = checksum(x, m);

    /* Check against reference */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");

    free(b);
    free(x);

    return TEST_SUMMARY();
}
