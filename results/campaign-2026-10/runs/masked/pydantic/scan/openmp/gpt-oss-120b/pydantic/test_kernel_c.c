#include <stdio.h>
#include <stdlib.h>
#include "kernel_c.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
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

    kernel_c(in, out, n);

    double chk = checksum(out, n);
    CHECK_CLOSE(chk, REF_CHECKSUM, REF_TOL, "checksum matches reference");

    free(in);
    free(out);
    return TEST_SUMMARY();
}
