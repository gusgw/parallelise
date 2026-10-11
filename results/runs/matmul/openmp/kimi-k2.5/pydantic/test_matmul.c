#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matmul.h"
#include "problem.h"
#include "reference.h"

int main(void)
{
    long n = REF_N;
    
    /* Allocate matrices */
    double *A = malloc(n * n * sizeof(double));
    double *B = malloc(n * n * sizeof(double));
    double *C = malloc(n * n * sizeof(double));
    
    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    /* Fill A and B using deterministic formulas from problem.h */
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }
    }
    
    /* Compute C = A * B */
    matmul(A, B, C, n);
    
    /* Compute checksum */
    double cs = checksum(C, n);
    
    /* Check against reference */
    double diff = fabs(cs - REF_CHECKSUM);
    
    printf("checksum = %.1f\n", cs);
    printf("expected = %.1f\n", REF_CHECKSUM);
    printf("diff     = %g\n", diff);
    printf("tol      = %g\n", REF_TOL);
    
    free(A);
    free(B);
    free(C);
    
    if (diff <= REF_TOL) {
        printf("PASS\n");
        return 0;
    } else {
        printf("FAIL\n");
        return 1;
    }
}
