#include "kernel_d.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"
#include <stdlib.h>
#include <string.h>

int main(void)
{
    long m = REF_M;
    int iters = (int)REF_ITERS;
    long n = m * m;
    
    /* Allocate arrays */
    double *x = (double *)malloc(n * sizeof(double));
    double *b = (double *)malloc(n * sizeof(double));
    
    if (!x || !b) {
        free(x);
        free(b);
        return 1;
    }
    
    /* Fill b using b_of from problem.h */
    for (long idx = 0; idx < n; idx++) {
        b[idx] = b_of(idx);
    }
    
    /* Call the kernel */
    cgsolve(x, b, m, iters);
    
    /* Compute checksum using checksum from problem.h
     * Note: checksum takes m (grid dimension), not n (total elements)
     */
    double cs = checksum(x, m);
    
    /* Check against reference */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
    
    free(x);
    free(b);
    
    return TEST_SUMMARY();
}
