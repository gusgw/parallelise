#include <stdio.h>
#include <stdlib.h>

#include "scan.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;
    double *in = (double *)malloc(sizeof(double) * n);
    double *out = (double *)malloc(sizeof(double) * n);
    if (!in || !out) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }
    for (long i = 0; i < n; ++i) {
        in[i] = in_of(i);
    }

    scan(in, out, n);

    double got = checksum(out, n);
    double want = REF_CHECKSUM;
    double tol = REF_TOL;
    CHECK_CLOSE(got, want, tol, "checksum");

    free(in);
    free(out);
    return TEST_SUMMARY();
}
