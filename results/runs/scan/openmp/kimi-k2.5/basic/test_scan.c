#include <stdio.h>
#include <stdlib.h>
#include "scan.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;
    
    /* Allocate input and output arrays */
    double *in = (double *)malloc(n * sizeof(double));
    double *out = (double *)malloc(n * sizeof(double));
    
    if (!in || !out) {
        printf("FAIL: memory allocation failed\n");
        return 1;
    }
    
    /* Fill input with deterministic values */
    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }
    
    /* Run the scan */
    scan(in, out, n);
    
    /* Compute checksum */
    double cs = checksum(out, n);
    
    /* Verify against reference */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum mismatch");
    
    free(in);
    free(out);
    
    return TEST_SUMMARY();
}
