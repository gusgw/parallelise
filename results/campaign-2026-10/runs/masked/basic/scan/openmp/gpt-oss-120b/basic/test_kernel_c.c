#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kernel_c.h"
#include "problem.h"
#include "reference.h"

int main(void) {
    long n = REF_N;
    double *in = (double*)malloc(n * sizeof(double));
    double *out = (double*)malloc(n * sizeof(double));
    if (!in || !out) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    for (long i = 0; i < n; ++i) {
        in[i] = in_of(i);
    }
    kernel_c(in, out, n);
    double chk = checksum(out, n);
    double diff = fabs(chk - REF_CHECKSUM);
    if (diff <= REF_TOL) {
        printf("PASS: checksum = %.15g, diff = %.3e\n", chk, diff);
        free(in);
        free(out);
        return 0;
    } else {
        printf("FAIL: checksum = %.15g, expected = %.15g, diff = %.3e\n", chk, (double)REF_CHECKSUM, diff);
        free(in);
        free(out);
        return 1;
    }
}
