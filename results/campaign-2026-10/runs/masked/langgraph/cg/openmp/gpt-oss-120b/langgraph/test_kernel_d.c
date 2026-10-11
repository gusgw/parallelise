#include <stdio.h>
#include <stdlib.h>
#include "kernel_d.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long m = REF_M;
    int iters = (int)REF_ITERS;
    long n = m * m;
    double *b = (double *)malloc(n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));
    if (!b || !x) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    for (long i = 0; i < n; ++i) {
        b[i] = b_of(i);
    }
    cgsolve(x, b, m, iters);
    double chk = checksum(x, m);
    CHECK_CLOSE(chk, REF_CHECKSUM, REF_TOL, "checksum");
    free(b);
    free(x);
    return TEST_SUMMARY();
}
