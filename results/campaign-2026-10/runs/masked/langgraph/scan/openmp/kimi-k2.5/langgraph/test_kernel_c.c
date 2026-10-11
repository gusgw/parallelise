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
    double *in = (double *)malloc(n * sizeof(double));
    double *out = (double *)malloc(n * sizeof(double));
    
    /* Fill input using problem.h helper */
    for (long i = 0; i < n; i++) {
        in[i] = in_of(i);
    }
    
    /* Run kernel */
    kernel_c(in, out, n);
    
    /* Compute checksum using problem.h helper */
    double cs = checksum(out, n);
    
    /* Check against reference */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum mismatch");
    
    free(in);
    free(out);
    
    return TEST_SUMMARY();
}
