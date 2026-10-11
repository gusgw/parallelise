#include "kernel_b.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"
#include <stdlib.h>

int main(void)
{
    long n = REF_N;
    int iters = (int)REF_ITERS;
    
    double *u = malloc(n * n * sizeof(double));
    if (!u) {
        printf("FAIL: memory allocation failed\n");
        return 1;
    }
    
    kernel_b(u, n, iters);
    
    double cs = checksum(u, n);
    
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum");
    
    free(u);
    
    return TEST_SUMMARY();
}
