#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "kernel_d.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(void)
{
    long m = REF_M;
    int iters = (int)REF_ITERS;
    long n = m * m;
    
    double *x = malloc(sizeof(double) * n);
    double *b = malloc(sizeof(double) * n);
    
    if (!x || !b) {
        fprintf(stderr, "Allocation failed\n");
        free(x);
        free(b);
        return 1;
    }
    
    /* Fill b using the problem definition */
    for (long t = 0; t < n; t++) {
        b[t] = b_of(t);
    }
    
    /* Run the solver */
    cgsolve(x, b, m, iters);
    
    /* Compute checksum - checksum() expects m (grid dimension), not n */
    double cs = checksum(x, m);
    
    /* Check against reference */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
    
    free(x);
    free(b);
    
    return TEST_SUMMARY();
}
