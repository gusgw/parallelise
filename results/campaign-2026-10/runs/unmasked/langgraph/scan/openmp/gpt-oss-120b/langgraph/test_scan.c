#include <stdlib.h>
#include <stdio.h>
#include "scan.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void) {
    long n = REF_N;
    double *in = (double *)malloc(n * sizeof(double));
    double *out = (double *)malloc(n * sizeof(double));
    if (!in || !out) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    for (long i = 0; i < n; ++i) {
        in[i] = in_of(i);
    }
    scan(in, out, n);
    double got = checksum(out, n);
    double want = REF_CHECKSUM;
    double tol = REF_TOL;
    CHECK_CLOSE(got, want, tol, "checksum mismatch");
    free(in);
    free(out);
    return TEST_SUMMARY();
}
