#include "cg.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"
#include <stdlib.h>

int main(void)
{
    long m = REF_M;
    int iters = REF_ITERS;
    long n = m * m;
    
    // Allocate and fill the right-hand side
    double *b = (double *)malloc(n * sizeof(double));
    for (long idx = 0; idx < n; idx++) {
        b[idx] = b_of(idx);
    }
    
    // Allocate solution vector
    double *x = (double *)malloc(n * sizeof(double));
    
    // Run the CG solver
    cgsolve(x, b, m, iters);
    
    // Compute checksum and compare to reference
    double cs = checksum(x, m);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "CG checksum");
    
    // Clean up
    free(x);
    free(b);
    
    return TEST_SUMMARY();
}
