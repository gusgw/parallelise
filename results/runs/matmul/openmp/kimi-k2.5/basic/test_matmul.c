#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "matmul.h"
#include "reference.h"
#include "problem.h"
#include "testing.h"

int main(void)
{
    long n = REF_N;
    
    /* Allocate matrices */
    double *A = malloc(n * n * sizeof(double));
    double *B = malloc(n * n * sizeof(double));
    double *C = malloc(n * n * sizeof(double));
    
    if (!A || !B || !C) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }
    
    /* Fill A and B using the deterministic fill functions from problem.h */
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }
    }
    
    /* Initialize C to zero (not strictly necessary but good practice) */
    memset(C, 0, n * n * sizeof(double));
    
    /* Compute C = A * B */
    matmul(A, B, C, n);
    
    /* Compute checksum */
    double cs = checksum(C, n);
    
    /* Verify against reference */
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum mismatch");
    
    /* Clean up */
    free(A);
    free(B);
    free(C);
    
    return TEST_SUMMARY();
}
