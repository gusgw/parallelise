#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    for (long idx = 0; idx < n; idx++) {
        b[idx] = b_of(idx);
    }
    
    /* Solve using GPU */
    cgsolve(x, b, m, iters);
    
    /* Compute checksum */
    double cs = checksum(x, m);
    
    /* Check against reference */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum mismatch with reference");
    
    free(x);
    free(b);
    
    return TEST_SUMMARY();
}
