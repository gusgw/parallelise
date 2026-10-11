#include <stdio.h>
#include <stdlib.h>
#include "jacobi.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;
    int iters = (int)REF_ITERS;
    
    double *u = malloc(n * n * sizeof(double));
    if (!u) {
        printf("FAIL: could not allocate memory\n");
        return 1;
    }
    
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }
    
    jacobi(u, n, iters);
    
    double cs = checksum(u, n);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum mismatch");
    
    free(u);
    
    return TEST_SUMMARY();
}
