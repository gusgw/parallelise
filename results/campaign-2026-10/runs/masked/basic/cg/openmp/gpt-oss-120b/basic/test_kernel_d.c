#include <stdlib.h>
#include <stdio.h>

#include "kernel_d.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long m = REF_M;
    long n = m * m;
    double *x = (double *)malloc(n * sizeof(double));
    double *b = (double *)malloc(n * sizeof(double));
    if (!x || !b) {
        printf("allocation failed\n");
        return 1;
    }
    for (long i = 0; i < n; ++i) {
        b[i] = b_of(i);
    }

    cgsolve(x, b, m, (int)REF_ITERS);

    double chk = checksum(x, m);
    CHECK_CLOSE(chk, REF_CHECKSUM, REF_TOL, "checksum matches reference");

    free(x);
    free(b);
    return TEST_SUMMARY();
}
