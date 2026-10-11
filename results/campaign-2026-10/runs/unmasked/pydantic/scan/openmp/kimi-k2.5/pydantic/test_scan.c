#include <stdio.h>
#include <stdlib.h>

#include "scan.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;
    double *in = malloc(sizeof(double) * n);
    double *out = malloc(sizeof(double) * n);
    
    if (!in || !out) {
        printf("FAIL: allocation failed\n");
        return 1;
    }

    /* Fill input array deterministically */
    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }

    /* Run the scan kernel */
    scan(in, out, n);

    /* Compute checksum and verify against reference */
    double cs = checksum(out, n);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum mismatch");

    free(in);
    free(out);

    return TEST_SUMMARY();
}
