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
    
    /* Allocate and fill b using the deterministic b_of function */
    double *b = malloc(n * sizeof(double));
    for (long idx = 0; idx < n; idx++) {
        b[idx] = b_of(idx);
    }
    
    /* Allocate solution vector x */
    double *x = malloc(n * sizeof(double));
    
    /* Run the conjugate gradient solver */
    cgsolve(x, b, m, iters);
    
    /* Compute checksum */
    double cs = checksum(x, m);
    
    /* Check against reference */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum mismatch");
    
    free(b);
    free(x);
    
    return TEST_SUMMARY();
}
