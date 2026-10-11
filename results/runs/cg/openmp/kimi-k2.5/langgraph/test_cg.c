#include "cg.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"
#include <stdlib.h>
#include <stdio.h>

int main(void)
{
    long m = REF_M;
    int iters = (int)REF_ITERS;
    long n = m * m;
    
    /* Allocate arrays */
    double *b = (double *)malloc(n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));
    
    if (!b || !x) {
        printf("FAIL: memory allocation failed\n");
        free(b);
        free(x);
        return 1;
    }
    
    /* Fill b with deterministic values */
    for (long idx = 0; idx < n; idx++) {
        b[idx] = b_of(idx);
    }
    
    /* Solve using conjugate gradient */
    cgsolve(x, b, m, iters);
    
    /* Compute checksum */
    double cs = checksum(x, m);
    
    /* Check against reference */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum mismatch");
    
    free(b);
    free(x);
    
    return TEST_SUMMARY();
}
