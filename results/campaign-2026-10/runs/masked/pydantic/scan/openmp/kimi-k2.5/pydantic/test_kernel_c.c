#include "kernel_c.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(void)
{
    long n = REF_N;
    
    /* Allocate input and output arrays */
    double *in = (double *)malloc(n * sizeof(double));
    double *out = (double *)malloc(n * sizeof(double));
    
    if (!in || !out) {
        printf("FAIL: Memory allocation failed\n");
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
    
    /* Check against reference */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum mismatch");
    
    free(in);
    free(out);
    
    return TEST_SUMMARY();
}
