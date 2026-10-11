#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "jacobi.h"
#include "reference.h"
#include "problem.h"

int main(void)
{
    long n = REF_N;
    int iters = (int)REF_ITERS;
    
    double *u = malloc(sizeof(double) * n * n);
    if (!u) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }
    
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            u[i * n + j] = u_init(i, j, n);
        }
    }
    
    jacobi(u, n, iters);
    
    double cs = checksum(u, n);
    double err = fabs(cs - REF_CHECKSUM);
    
    printf("checksum = %.16f\n", cs);
    printf("expected = %.16f\n", REF_CHECKSUM);
    printf("error    = %.16f\n", err);
    printf("tol      = %.16f\n", REF_TOL);
    
    free(u);
    
    if (err > REF_TOL) {
        printf("FAIL: error exceeds tolerance\n");
        return 1;
    }
    
    printf("PASS\n");
    return 0;
}
