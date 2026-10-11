#include <stdio.h>
#include <stdlib.h>

#include "kernel_d.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long m = REF_M;
    int iters = (int)REF_ITERS;
    long n = m * m;
    
    double *x = malloc(sizeof(double) * n);
    double *b = malloc(sizeof(double) * n);
    
    if (!x || !b) {
        printf("FAIL: allocation failed\n");
        free(x);
        free(b);
        return 1;
    }
    
    for (long t = 0; t < n; t++) {
        b[t] = b_of(t);
    }
    
    cgsolve(x, b, m, iters);
    
    double cs = checksum(x, m);
    
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
    
    free(x);
    free(b);
    
    return TEST_SUMMARY();
}
