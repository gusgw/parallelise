#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "cg.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long m = REF_M;
    int iters = (int)REF_ITERS;
    long n = m * m;

    double *x = (double *)malloc(sizeof(double) * n);
    double *b = (double *)malloc(sizeof(double) * n);
    if (!x || !b) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    for (long i = 0; i < n; ++i) {
        b[i] = b_of(i);
    }

    cgsolve(x, b, m, iters);

    double cs = checksum(x, m);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum matches reference");

    free(x);
    free(b);
    return TEST_SUMMARY();
}
