/* test_cg.c - Unit test for cg.cu */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cg.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long m = REF_M;
    int iters = (int)REF_ITERS;
    long n = m * m;
    
    double *x = (double *)malloc(n * sizeof(double));
    double *b = (double *)malloc(n * sizeof(double));
    
    if (!x || !b) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }
    
    /* Fill b deterministically */
    for (long t = 0; t < n; t++) {
        b[t] = b_of(t);
    }
    
    /* Solve using GPU */
    cgsolve(x, b, m, iters);
    
    /* Compute checksum */
    double cs = checksum(x, m);
    
    /* Check against reference */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
    
    free(x);
    free(b);
    
    return TEST_SUMMARY();
}
