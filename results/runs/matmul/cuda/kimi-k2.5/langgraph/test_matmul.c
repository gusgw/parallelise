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
    double *A = malloc(sizeof(double) * n * n);
    double *B = malloc(sizeof(double) * n * n);
    double *C = malloc(sizeof(double) * n * n);

    if (!A || !B || !C) {
        fprintf(stderr, "malloc failed\n");
        free(A);
        free(B);
        free(C);
        return 1;
    }

    /* Fill A and B using deterministic formulas from problem.h */
    for (long i = 0; i < n; i++) {
        for (long j = 0; j < n; j++) {
            A[i * n + j] = A_of(i, j);
            B[i * n + j] = B_of(i, j);
        }
    }

    /* Initialize C to zeros */
    memset(C, 0, sizeof(double) * n * n);

    /* Compute C = A * B on GPU */
    matmul(A, B, C, n);

    /* Compute checksum and verify against reference */
    double cs = checksum(C, n);
    CHECK_CLOSE(cs, REF_CHECKSUM, REF_TOL, "checksum matches reference");

    free(A);
    free(B);
    free(C);

    return TEST_SUMMARY();
}
