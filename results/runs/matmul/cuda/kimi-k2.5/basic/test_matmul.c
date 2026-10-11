#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matmul.h"
#include "problem.h"
#include "reference.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;
    double *A = (double *)malloc(sizeof(double) * n * n);
    double *B = (double *)malloc(sizeof(double) * n * n);
    double *C = (double *)malloc(sizeof(double) * n * n);
    
    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }
    
    /* Fill matrices A and B using deterministic fill functions */
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }
    }
    
    /* Initialize C to zeros */
    memset(C, 0, sizeof(double) * n * n);
    
    /* Perform matrix multiplication on GPU */
    matmul(A, B, C, n);
    
    /* Compute checksum */
    double cs = checksum(C, n);
    
    /* Verify against reference */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum mismatch");
    
    free(A);
    free(B);
    free(C);
    
    return TEST_SUMMARY();
}
