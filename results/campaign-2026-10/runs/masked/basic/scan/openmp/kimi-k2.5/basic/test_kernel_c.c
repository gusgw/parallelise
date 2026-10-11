#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "kernel_c.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;
    double *in = malloc(sizeof(double) * n);
    double *out = malloc(sizeof(double) * n);
    
    if (!in || !out) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    /* Fill input array using in_of from problem.h */
    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }

    /* Run the kernel */
    kernel_c(in, out, n);

    /* Compute checksum using checksum from problem.h */
    double cs = checksum(out, n);

    /* Verify against reference */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum mismatch");

    free(in);
    free(out);

    return TEST_SUMMARY();
}
