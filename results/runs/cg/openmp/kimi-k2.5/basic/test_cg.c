#include "cg.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"
#include <stdlib.h>

int main(void)
{
    long m = REF_M;
    int iters = (int)REF_ITERS;
    long n = m * m;
    
    /* Allocate arrays */
    double *b = (double *)malloc(n * sizeof(double));
    double *x = (double *)malloc(n * sizeof(double));
    
    if (!b || !x) {
        CHECK(0, "memory allocation failed");
        free(b);
        free(x);
        return TEST_SUMMARY();
    }
    
    /* Fill b with deterministic values */
    for (long idx = 0; idx < n; idx++) {
        b[idx] = b_of(idx);
    }
    
    /* Run the conjugate gradient solver */
    cgsolve(x, b, m, iters);
    
    /* Compute checksum and verify against reference */
    double cs = checksum(x, m);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum mismatch");
    
    free(b);
    free(x);
    
    return TEST_SUMMARY();
}
