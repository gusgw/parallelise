#include <stdio.h>
#include <stdlib.h>
#include "kernel_d.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long m = REF_M;
    long n = m * m;
    double *x = (double*)malloc(n * sizeof(double));
    if (!x) return 1;
    double *bvec = (double*)malloc(n * sizeof(double));
    if (!bvec) { free(x); return 1; }
    for (long i = 0; i < n; ++i) {
        bvec[i] = b_of(i);
    }
    // solve
    cgsolve(x, bvec, m, REF_ITERS);
    // compute checksum
    double chk = checksum(x, m);
    CHECK_CLOSE(chk, REF_CHECKSUM, REF_TOL, "checksum");
    free(x);
    free(bvec);
    return TEST_SUMMARY();
}
